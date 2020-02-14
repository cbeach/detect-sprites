from collections import defaultdict
import itertools
from math import ceil
import pickle
import sys
from time import time

import cv2
import numpy as np

from .patch import Node, Patch
from .sprite_util import show_images, show_image, get_frame, neighboring_points, sort_colors, conjugate_numbers
from .db.data_store import DataStore
from .db.models import EdgeM, GameM, NodeM, PatchM #, FrameGraphM

class FrameGraph:
    @staticmethod
    def from_raw_frame(game, play_number, frame_number, bg_color=None, indirect=True, ds=None):
        img = get_frame(game, play_number, frame_number)
        return FrameGraph(img, game=game, play_num=play_number, frame_num=frame_number, bg_color=bg_color | img[0][0], indirect=indirect, ds=ds)

    @staticmethod
    def from_path(path, game, bg_color=None, indirect=True, ds=None):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return FrameGraph(img, game=game, bg_color=bg_color, indirect=indirect, ds=ds)

    def __init__(self, frame=None, game='SuperMarioBros-Nes', bg_color=None, indirect=True, graph=None, subgraph=None, ds=None, play_num=0, frame_num=0, init_alpha=False):
        sess = ds.Session()
        self.game = sess.query(GameM).filter(GameM.name==game).one()
        self.game_id = self.game.id
        self.play_num = play_num
        self.frame_num = frame_num

        if frame is not None:
            self._init_frame(frame, bg_color, indirect)
            self._init_graph()
        elif subgraph is not None and graph is not None:
            self._from_subgraph(graph, subgraph)
        else:
            raise ValueError(f'unacceptable inputs: frame: {frame}, subgraph: {subgraph}, graph: {graph}')

    def _init_frame(self, frame, bg_color, indirect, init_alpha=False):
        # Frame init
        if frame.shape[-1] == 3:
            self.raw_frame = frame
            self.alpha_chan = np.full(frame.shape[:2], 255, dtype='uint8')
            self._init_alpha = True
        elif frame.shape[-1] == 4:
            self.raw_frame = frame[:, :, :3]
            self.alpha_chan = frame[:, :, 3]
            self._init_alpha = False

        self._init_alpha = self._init_alpha | init_alpha
        self.indirect = indirect

        if bg_color is not None:
            self.bg_color = np.array(bg_color[:3], dtype='uint8')
        else:
            self.bg_color = None

        self.patch_colors = []
        self.bounding_boxes = []
        self.parse_frame()
        self.patch_index = defaultdict(list)
        self.pix_index = {}
        self.index_patches()

    def _init_graph(self):
        # graph init
        self.offset_hash_to_patch = {i.offset_hash():i for i in self.patches}

        self.hashes = sorted(list(set([hash(i) for i in self.patches])))
        self.offset_hashes = sorted([i.offset_hash() for i in self.patches])

        self.hash_to_patches = {hash(i[0]):i for i in itertools.groupby(self.patches, key=hash)}
        self.offset_hash_to_index = {p.offset_hash():i for i, p in enumerate(self.patches)}
        self.graph = self.build_graph()

        for i in self.patches:
            if (i.bounding_box[0][0] == 0 or i.bounding_box[0][1] == 0
                or i.bounding_box[1][0] == self.raw_frame.shape[0]
                or i.bounding_box[1][1] == self.raw_frame.shape[1]):
                i.mark_as_frame_edge()

            for x, y in i.get_neighboring_patch_pixels(self.raw_frame):
                if self.is_background(x, y):
                    i.mark_as_bg_edge()
                    break

        self._subgraphs = [Graphlet(self, sg) for sg in self._isolate_offset_subgraphs()]

    def remove_nodes(self, nodes):
        img1 = self.raw_frame
        for p in self.patches:
            img1 = p.fill_patch(img1)

        ohh = {n.oh():n for n in nodes}
        dead_patches = [i for i, _ in enumerate(self.offset_hashes) if i not in ohh]

        current_rows = list(self.offset_adjacency_matrix)
        rows = []
        ind = 0
        for i, cur in enumerate(current_rows):
            if i != dead_patches[ind]:
                rows.append(cur)

        current_cols = list(np.array(rows, dtype=bool).T)
        cols = []
        ind = 0
        for i, cur in enumerate(current_cols):
            if i != dead_patches[ind]:
                cols.append(cur)
        new_patches = np.array(cols, dtype=bool).T

        backup = self.patches
        self.patches = list(filter(lambda k: k.oh() not in ohh, self.patches))
        not_patches = list(filter(lambda k: k.oh() in ohh, backup))
        self.offset_hashes = list(filter(lambda k: k not in ohh, self.offset_hashes))

        img2 = self.raw_frame
        for p in self.patches:
            img2 = p.fill_patch(img2)
        for p in not_patches:
            img2 = p.fill_patch(img2, color=(255, 0, 0))


        show_image(img1, scale=4.0)
        show_image(img2, scale=4.0)

    def parse_frame(self):
        frame = self.raw_frame
        mask = np.zeros(frame.shape[:-1])
        size = frame.shape[0] * frame.shape[1]
        self.patches = []
        self.bounding_boxes = []

        patches = []
        palette = {}
        patch_colors = []
        for i, row in enumerate(mask):
            for j, pix in enumerate(row):
                palette[tuple(frame[i][j])] = True
                if pix == False:
                    mask[i][j] = True
                    if self.is_background(i, j):
                        continue
                    node = Node(frame, i, j, self.play_num, self.frame_num, mask=mask, indirect=self.indirect)

                    for x, y in node.coord_list():
                        mask[x][y] = True

                    self.patches.append(node)
                    self.bounding_boxes.append(node.bounding_box)
                    patch_colors.append(tuple(frame[i][j]))

        self.patches = sorted(self.patches, key=lambda p: p.offset_hash())
        self.palette = sort_colors([(int(i[0]), int(i[1]), int(i[2])) for i in palette.keys()])
        self.patch_colors = [self.color_as_palette(i) for i in patch_colors]
        if self._init_alpha is True:
            self._init_alpha = False
            self.alpha_chan = np.zeros_like(mask, dtype=np.uint8)
            for node in self.patches:
                for x, y in node.coord_list():
                    self.alpha_chan[x][y] = 255

    def build_graph(self):
        """
            For each patch, get the list of pixels just outside the patch's outer edge.
            Use that list of pixels to find all patches that are touching this one.
        """
        self.offset_adjacency_matrix = np.zeros((len(self.offset_hashes), len(self.offset_hashes)), dtype=bool)
        for i, patch in enumerate(self.patches):
            current_hash = hash(patch)
            current_offset_hash = patch.offset_hash()
            nbr_pixels = patch.get_neighboring_patch_pixels(self.raw_frame)
            nbr_patches = list(self.get_patches_by_coord(nbr_pixels))
            patch.neighbors.extend(nbr_patches)
            patch.cull_neighbors()
            for npatch in nbr_patches:
                npatch_hash = hash(npatch)
                npatch_offset_hash = npatch.offset_hash()
                self.offset_adjacency_matrix[self.offset_hash_to_index[current_offset_hash]][self.offset_hash_to_index[npatch_offset_hash]] = True

    def get_adjacent_patches(self, p_hash=None, o_hash=None):
        if p_hash is not None:
            raise NotImplementedError('get_adjacent_patches is not implemented for p_hash')
        elif o_hash is not None:
            primary_index = self.offset_hash_to_index[o_hash]
            adj_mat = self.offset_adjacency_matrix

        adj_nodes = np.flatnonzero(adj_mat[primary_index])
        return [self.frame.patches[i] for i in adj_nodes]

    def _isolate_offset_subgraphs(self):
        adj = np.nonzero(self.offset_adjacency_matrix)
        adj_coords = {i[0]:[j[1] for j in i[1]] for i in itertools.groupby(zip(adj[0], adj[1]), key=lambda x: x[0])}
        subgraphs = [[self.offset_hash_to_patch[self.offset_hashes[i]]] for i in range(len(self.offset_hashes)) if i not in adj_coords]
        o_subgraphs = [[self.offset_hashes[i]] for i in range(len(self.offset_hashes)) if i not in adj_coords]
        visited = np.zeros(self.offset_adjacency_matrix.shape, dtype=bool)

        while len(adj_coords) > 0:
            stack = [list(adj_coords.keys())[0]]
            connected = []
            ohs = []
            while len(stack) > 0:
                curr = stack.pop()
                if adj_coords.get(curr, None) is None:
                    continue

                for i in adj_coords[curr]:
                    if visited[curr][i] == False:
                        stack.append(i)
                        visited[curr][i] = True
                    else:
                        continue

                connected.append(self.offset_hash_to_patch[self.offset_hashes[curr]])
                ohs.append(self.offset_hashes[curr])
                del adj_coords[curr]

            subgraphs.append(connected)
            o_subgraphs.append(ohs)

        return subgraphs

    def subgraphs(self):
        return self._subgraphs

    def subgraph_area(self, subgraph):
        return sum(map(lambda p: p.area(), subgraph))

    def color_as_palette(self, color):
        return self.palette.index(color)

    def get_patches_by_coord(self, coords):
        try:
            iter(coords)
        except TypeError:
            return [self.pix_index[coords]]

        background_filtered_out = [c for c in coords if not self.is_background(c[0], c[1])]
        return [self.pix_index[c] for c in background_filtered_out]

    def index_patches(self):
        for patch in self.patches:
            patch_hash = hash(patch)
            self.patch_index[patch_hash].append(patch)

            for coord in patch.coord_list():
                self.pix_index[coord] = patch

    def is_background(self, x, y):
        if self.bg_color is None and self._init_alpha is True:
            return False
        elif self.bg_color is None and self._init_alpha is False:
            return self.alpha_chan[x][y] == 0

        return np.array_equal(self.bg_color[:3], self.raw_frame[x][y][:3]) is True

    def add_neighbors_to_nodes(self):
        rf = self.raw_frame.copy()
        for i, patch in enumerate(self.patches):
            phash = hash(patch)
            ohash = patch.offset_hash()
            am_row = self.offset_adjacency_matrix[self.offset_hash_to_index[ohash]]
            nbrs = [self.patches[i] for i, j in enumerate(am_row) if j == 1]
            patch.neighbors.extend(nbrs)

    def store(self, ds=None):
        if ds is None:
            ds = self.ds

        sess = ds.Session()

        nodes = {node.offset_hash():(node, node.store(game=self.game, play_number=self.play_num, frame_number=self.frame_num, commit=False, ds=ds, sess=sess)) for node in self.patches}

        for node, model in nodes.values():
            sess.add(model)
        sess.commit()

        # store edges
        edges = []
        for node, model in nodes.values():
            bb1 = node.bounding_box
            for nbr in node.neighbors:
                bb2 = nbr.bounding_box
                x_o = bb1[0][0] - bb2[0][0]
                y_o = bb1[0][1] - bb2[0][1]
                edges.append(EdgeM(left_id=model.id, right_id=nodes[nbr.offset_hash()][1].id, x_offset=x_o, y_offset=y_o))
        sess.add_all(edges)
        sess.commit()

    def __eq__(self, other):
        sorted(self.offset_hashes) == sorted(other.offset_hashes) and np.array_equal(self.offset_adjacency_matrix, other.offset_adjacency_matrix)

    def normalize_image(self):
        sx, sy, sc = self.raw_frame.shape
        hash_mask = [[None] * sy] * sx
        for coord, patch in self.pix_index.items():
            hash_mask[coord[0]][coord[1]] = hash(patch)
        return hash_mask

    # --- Debugging ---
    def print_offset_adjacency_matrix(self):
        print()
        print(self.offset_adjacency_matrix.shape)
        adm_str = '-' * (self.offset_adjacency_matrix.shape[0] + 2) + '\n'
        for i in self.offset_adjacency_matrix:
            row_str = '|'
            for j in i:
                if j != 0:
                    row_str += 'x'
                else:
                    row_str += ' '
            adm_str += f'{row_str}|\n'
        adm_str += '-' * (self.offset_adjacency_matrix.shape[0] + 2)
        print(adm_str)
        print()

    def show(self, scale=1.0):
        show_image(self.raw_frame, scale=scale)

    def show_neighbors(self):
        pass


class Graphlet:
    def __init__(self, graph, patches, ds=None):
        if ds is not None:
            self.ds = ds
        else:
            self.ds = ds = DataStore(games_path='./games.json')

        self.graph = graph
        self.bg_color = graph.bg_color
        self.nodes = patches

        self.bb = self._bounding_box()
        self.palette = sort_colors(set([tuple(i.color[:3]) for i in self.nodes]))
        if graph.bg_color is not None:
            self.palette = self.palette + [tuple(graph.bg_color[:3])]
            self.bg_color = self.color_as_palette(tuple(graph.bg_color[:3]))
        else:
            self.bg_color = None
        self.clipped_frame = graph.raw_frame[self.bb[0][0]:self.bb[1][0], self.bb[0][1]:self.bb[1][1],:]
        self.mask = self._mask()
        #self.patch_mask = self._patch_mask()
        self.hash_mask = self._hash_mask()
        self.palettized = self._palettize()

    def color_as_palette(self, color):
        return self.palette.index(tuple(color))

    def area(self):
        return sum([node.area() for node in self.nodes])

    def _bounding_box(self):
        xs = []
        ys = []
        bbs = []
        for i in self.nodes:
            xs.append(i.bounding_box[0][0])
            xs.append(i.bounding_box[1][0])
            ys.append(i.bounding_box[0][1])
            ys.append(i.bounding_box[1][1])
            bbs.append(i.bounding_box)
        return ((min(xs), min(ys)), (max(xs), max(ys)))

    def _mask(self):
        left_most = list(filter(lambda p: p.bounding_box[0][1] == self.bb[0][1], self.nodes))
        top_left_patch = list(sorted(left_most, key=lambda p: p.bounding_box[0][0]))[0]
        tlpbb = top_left_patch.bounding_box
        xt, yt = (tlpbb[0][0] - self.bb[0][0], tlpbb[0][1] - self.bb[0][1])

        shape = self.bb[1][0] - self.bb[0][0], self.bb[1][1] - self.bb[0][1]
        mask = np.zeros(shape, dtype=bool)

        for x, y in top_left_patch.patch._patch.translate(xt, yt):
            mask[x][y] = True

        for i in self.nodes:
            ibb = i.bounding_box
            xt, yt = (ibb[0][0] - self.bb[0][0], ibb[0][1] - self.bb[0][1])
            for x, y in i.patch._patch.translate(xt, yt):
                mask[x][y] = True
        return mask

    def _patch_mask(self):
        mask = np.zeros(self.mask.shape, dtype='uint16')
        bblx, bbly = self.bounding_box[0]
        for i, node in  enumerate(self.nodes):
            for x, y in [(x - bblx, y - bbly) for x, y in node.coord_list()]:
                mask[x][y] = i + 1

    def _hash_mask(self):
        mask = np.zeros(self.mask.shape, dtype=str)
        bblx, bbly = self.bb[0]
        for i, node in  enumerate(self.nodes):
            for x, y in [(x - bblx, y - bbly) for x, y in node.coord_list()]:
                mask[x][y] = str(hash(node))

    def _palettize(self):
        temp = np.zeros(self.mask.shape, dtype='uint8')
        for x, row in enumerate(temp):
            for y, pixel in enumerate(row):
                if self.mask[x][y]:
                    temp[x][y] = self.color_as_palette(self.clipped_frame[x][y])
        return temp

    def _un_palettize(self, mask):
        temp = np.zeros(self.clipped_frame.shape, dtype='uint8')
        for x, row in enumerate(temp):
            for y, pixel in enumerate(row):
                if self.mask[x][y]:
                    temp[x][y] = self.graph.palette[self.palettized[x][y]]
        return temp

    def to_image(self, border=1):
        xs = []
        ys = []
        bbs = []
        for i in self.nodes:
            xs.append(i.bounding_box[0][0])
            xs.append(i.bounding_box[1][0])
            ys.append(i.bounding_box[0][1])
            ys.append(i.bounding_box[1][1])
            bbs.append(i.bounding_box)
        bb = ((min(xs), min(ys)), (max(xs), max(ys)))

        left_most = list(filter(lambda p: p.bounding_box[0][1] == bb[0][1], self.nodes))
        top_left_patch = list(sorted(left_most, key=lambda p: p.bounding_box[0][0]))[0]
        tlpbb = top_left_patch.bounding_box
        xt, yt = (tlpbb[0][0] - bb[0][0], tlpbb[0][1] - bb[0][1])

        shape = bb[1][0] - bb[0][0], bb[1][1] - bb[0][1], 3
        frame = np.zeros(shape, dtype='uint8')

        for x, y in top_left_patch.patch._patch.translate(xt, yt):
            frame[x][y] = top_left_patch.color

        for i in self.nodes:
            ibb = i.bounding_box
            xt, yt = (ibb[0][0] - bb[0][0], ibb[0][1] - bb[0][1])
            for x, y in i.patch._patch.translate(xt, yt):
                frame[x][y] = i.color

        img = frame
        if border > 0:
            border_color=[0, 0, 0]
            sx, sy = frame.shape[0] + border * 2, frame.shape[1] + border * 2
            img = np.array([[border_color] * sy] * sx, dtype='uint8')
            img[border:sx - border, border:sy - border, :] = frame[:, :, :]

        return img
    def show(self, border=1, parent=None):
        img = self.to_image(border)
        if parent is not None:
            return show_images((self.fill(parent), img), scale=3)
        else:
            return show_image(img, scale=3)

    def fill(self, frame=None):
        if frame is None:
            frame = self.frame.raw_frame.copy()
        for i in self.nodes:
            frame = i.fill_patch(frame)
        return frame

    def ask_if_sprite(self, bg_color=None, parent_img=None, scale=8.0):
        print('> is this subgraph a sprite [y/N]?')
        # Create a new sprite if yes
        resp =  self.show(parent=parent_img)
        cv2.destroyAllWindows()
        return resp

    def touches_edge(self):
        return any([n.touches_edge() for n in self.nodes])

    def __eq__(self, other):
        return np.array_equal(self.hash_mask, other.hash_mask)
