from collections import defaultdict
import itertools
from math import ceil
import pickle
import sys
from time import time

import cv2
import numpy as np

from patch import Node
from sprite_util import show_image, get_frame, neighboring_points, sort_colors
from db.data_store import DataStore
from db.models import EdgeM, GameM, NodeM, PatchM #, FrameGraphM

class FrameGraph:
    @staticmethod
    def from_raw_frame(game, play_number, frame_number, bg_color=None, indirect=True, ds=None):
        return FrameGraph(get_frame(game, play_number, frame_number), game=game, play_num=play_number, frame_num=frame_number, bg_color=bg_color, indirect=indirect, ds=ds)

    def __init__(self, frame=None, game='SuperMarioBros-Nes', bg_color=None, indirect=True, graph=None, subgraph=None, ds=None, play_num=0, frame_num=0):
        #if ds is not None:
        #    self.ds = ds
        #else:
        #    self.ds = ds = DataStore(games_path='./games.json')

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

    def _init_frame(self, frame, bg_color, indirect):
        # Frame init
        self.raw_frame = frame
        self.indirect = indirect

        if bg_color is None:
            self.bg_color = self.raw_frame[0][0].copy()
        else:
            self.bg_color = bg_color

        self.patch_colors = []
        self.bounding_boxes = []
        self.parse_frame()
        self.patch_index = defaultdict(list)
        self.pix_index = {}
        self.index_patches()

    def _init_graph(self):
        # graph init
        start = time()
        self.hash_to_patch = {hash(i):i for i in self.patches}
        self.offset_hash_to_patch = {i.offset_hash():i for i in self.patches}

        start = time()
        self.hashes = sorted(list(set([hash(i) for i in self.patches])))
        self.offset_hashes = sorted([i.offset_hash() for i in self.patches])

        start = time()
        self.hash_to_index = {self.hashes[i]:i for i in range(len(self.hashes))}
        self.offset_hash_to_index = {self.offset_hashes[i]:i for i in range(len(self.offset_hashes))}

        self.index_to_hash = {i:self.hashes[i] for i in range(len(self.hashes))}
        self.index_to_offset_hash = {i:self.offset_hashes[i] for i in range(len(self.offset_hashes))}

        start = time()
        self.offset_hash_to_hash = {i.offset_hash():hash(i) for i in self.patches}

        self.hash_to_offset_hash = {}
        for i in self.patches:
            p_hash = hash(i)
            o_hash = i.offset_hash()
            if self.hash_to_offset_hash.get(p_hash, None) is None:
                self.hash_to_offset_hash[p_hash] = [o_hash]
            else:
                self.hash_to_offset_hash[p_hash].append(o_hash)

        start = time()
        self.adjacency_matrix = np.zeros((len(self.hashes), len(self.hashes)), dtype=bool)
        self.offset_adjacency_matrix = np.zeros((len(self.offset_hashes), len(self.offset_hashes)), dtype=bool)

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

        self._subgraphs = [PatchGraph(self, sg) for sg in self._isolate_offset_subgraphs()]

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
                    node = Node(frame, i, j, mask=mask, indirect=self.indirect)

                    for x, y in node.coord_list():
                        mask[x][y] = True

                    self.patches.append(node)
                    self.bounding_boxes.append(node.bounding_box)
                    patch_colors.append(tuple(frame[i][j]))

        self.palette = sort_colors([(int(i[0]), int(i[1]), int(i[2])) for i in palette.keys()])
        self.patch_colors = [self.color_as_palette(i) for i in patch_colors]

    def build_graph(self):
        """
            For each patch, get the list of pixels just outside the patch's outer edge.
            Use that list of pixels to find all patches that are touching this one.
        """
        for patch in self.patches:
            current_hash = hash(patch)
            current_offset_hash = patch.offset_hash()
            nbr_pixels = patch.get_neighboring_patch_pixels(self.raw_frame)
            nbr_patches = list(set(self.get_patches_by_coord(nbr_pixels)))
            for npatch in nbr_patches:
                npatch_hash = hash(npatch)
                npatch_offset_hash = npatch.offset_hash()
                self.adjacency_matrix[self.hash_to_index[current_hash]][self.hash_to_index[npatch_hash]] = True
                self.offset_adjacency_matrix[self.offset_hash_to_index[current_offset_hash]][self.offset_hash_to_index[npatch_offset_hash]] = True

    def get_adjacent_patches(self, p_hash=None, o_hash=None):
        if p_hash is not None:
            primary_index = self.hash_to_index[p_hash]
            adj_mat = self.adjacency_matrix
        elif o_hash is not None:
            primary_index = self.offset_hash_to_index[o_hash]
            adj_mat = self.offset_adjacency_matrix

        adj_nodes = np.flatnonzero(adj_mat[primary_index])
        return [self.frame.patches[i] for i in adj_nodes]

    def _isolate_offset_subgraphs(self):
        adj = np.nonzero(self.offset_adjacency_matrix)
        adj_coords = {i[0]:[j[1] for j in i[1]] for i in itertools.groupby(zip(adj[0], adj[1]), key=lambda x: x[0])}
        subgraphs = [[self.offset_hash_to_patch[self.index_to_offset_hash[i]]] for i in range(len(self.offset_hashes)) if i not in adj_coords]
        o_subgraphs = [[self.index_to_offset_hash[i]] for i in range(len(self.offset_hashes)) if i not in adj_coords]
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

                connected.append(self.offset_hash_to_patch[self.index_to_offset_hash[curr]])
                ohs.append(self.index_to_offset_hash[curr])
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
        return np.array_equal(self.bg_color, self.raw_frame[x][y]) is True

    def add_neighbors_to_nodes(self):
        for i, patch in enumerate(self.patches):
            phash = hash(patch)
            ohash = patch.offset_hash()
            sparse_nbrs = self.offset_adjacency_matrix[self.offset_hash_to_index[ohash]]
            nbr_indices = np.nonzero(sparse_nbrs)
            for j in nbr_indices[0]:
                patch.neighbors.append(self.patches[j])

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


            #left_id = Column(Integer, ForeignKey('nodes.id'), primary_key=True)
            #right_id = Column(Integer, ForeignKey('nodes.id'), primary_key=True)
            #x_offset = Column(Integer)
            #y_offset = Column(Integer)


    # --- Debugging ---
    def print_adjacency_matrix(self):
        print()
        print(self.adjacency_matrix.shape)
        adm_str = '-' * (self.adjacency_matrix.shape[0] + 2) + '\n'
        for i in self.adjacency_matrix:
            row_str = '|'
            for j in i:
                if j != 0:
                    row_str += 'x'
                else:
                    row_str += ' '
            adm_str += f'{row_str}|\n'
        adm_str += '-' * (self.adjacency_matrix.shape[0] + 2)
        print(adm_str)
        print()

    def show(self, scale=None):
        show_image(self.raw_frame, scale=scale)



class PatchGraph:
    def __init__(self, graph, patches, ds=None):
        if ds is not None:
            self.ds = ds
        else:
            self.ds = ds = DataStore(games_path='./games.json')

        self.graph = graph
        self.bg_color = graph.bg_color
        self.subgraph = patches

        self.bb = self._bounding_box()
        self.palette = sort_colors(set([tuple(i.color) for i in self.subgraph]))
        self.clipped_frame = graph.raw_frame[self.bb[0][0]:self.bb[1][0], self.bb[0][1]:self.bb[1][1],:]
        self.hash_list = sorted([hash(i) for i in self.subgraph])
        self.mask = self._mask()
        self.palettized = self._palettize()

    def color_as_palette(self, color):
        return self.palette.index(tuple(color))

    def _bounding_box(self):
        xs = []
        ys = []
        bbs = []
        for i in self.subgraph:
            xs.append(i.bounding_box[0][0])
            xs.append(i.bounding_box[1][0])
            ys.append(i.bounding_box[0][1])
            ys.append(i.bounding_box[1][1])
            bbs.append(i.bounding_box)
        return ((min(xs), min(ys)), (max(xs), max(ys)))

    def _mask(self):
        left_most = list(filter(lambda p: p.bounding_box[0][1] == self.bb[0][1], self.subgraph))
        top_left_patch = list(sorted(left_most, key=lambda p: p.bounding_box[0][0]))[0]
        tlpbb = top_left_patch.bounding_box
        xt, yt = (tlpbb[0][0] - self.bb[0][0], tlpbb[0][1] - self.bb[0][1])

        shape = self.bb[1][0] - self.bb[0][0], self.bb[1][1] - self.bb[0][1]
        mask = np.zeros(shape, dtype=bool)

        for x, y in top_left_patch.patch._patch.translate(xt, yt):
            mask[x][y] = True

        for i in self.subgraph:
            ibb = i.bounding_box
            xt, yt = (ibb[0][0] - self.bb[0][0], ibb[0][1] - self.bb[0][1])
            for x, y in i.patch._patch.translate(xt, yt):
                mask[x][y] = True
        return mask

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

    def show(self):
        xs = []
        ys = []
        bbs = []
        for i in self.subgraph:
            xs.append(i.bounding_box[0][0])
            xs.append(i.bounding_box[1][0])
            ys.append(i.bounding_box[0][1])
            ys.append(i.bounding_box[1][1])
            bbs.append(i.bounding_box)
        bb = ((min(xs), min(ys)), (max(xs), max(ys)))

        left_most = list(filter(lambda p: p.bounding_box[0][1] == bb[0][1], self.subgraph))
        top_left_patch = list(sorted(left_most, key=lambda p: p.bounding_box[0][0]))[0]
        tlpbb = top_left_patch.bounding_box
        xt, yt = (tlpbb[0][0] - bb[0][0], tlpbb[0][1] - bb[0][1])

        shape = bb[1][0] - bb[0][0], bb[1][1] - bb[0][1], 3
        frame = np.zeros(shape, dtype='uint8')

        for x, y in top_left_patch.patch._patch.translate(xt, yt):
            frame[x][y] = top_left_patch.color

        for i in self.subgraph:
            ibb = i.bounding_box
            xt, yt = (ibb[0][0] - bb[0][0], ibb[0][1] - bb[0][1])
            for x, y in i.patch._patch.translate(xt, yt):
                frame[x][y] = i.color

        return show_image(frame, scale=3)

    def fill(self, frame=None):
        if frame is None:
            frame = self.frame.raw_frame.copy()
        for i in self.subgraph:
            frame = i.fill_patch(frame)
        return frame

    def ask_if_sprite(self, bg_color=None):
        print('> is this subgraph a sprite [y/N]?')
        # Create a new sprite if yes
        return self.show(), Sprite(self)

    def __eq__(self, other):
        return self.hash_list == other.hash_list


class Sprite:
    def __init__(self, patch_graph, palette=None):
        pass

