from collections import defaultdict
import itertools
import pickle
import sys
from time import time

import cv2
import numpy as np

from patch import Node
from sprite_util import show_image, get_frame, neighboring_points

class PatchGraph:
    @staticmethod
    def from_raw_frame(game, play_number, frame_number, bg_color=None, indirect=True):
        return PatchGraph(get_frame(game, play_number, frame_number), bg_color=bg_color, indirect=indirect)

    def __init__(self, frame, game='SuperMarioBros-Nes', bg_color=None, indirect=True):
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
                or i.bounding_box[1][0] == frame.shape[0]
                or i.bounding_box[1][1] == frame.shape[1]):
                i.mark_as_frame_edge()

            for x, y in i.get_neighboring_patch_pixels(self.raw_frame):
                if self.is_background(x, y):
                    i.mark_as_bg_edge()
                    break

        temp = self.raw_frame.copy()
        for i in self.patches:
            if i.bg_edge is True and i.frame_edge is True:
                temp = i.fill_patch(temp, color=(255, 0, 255))
            elif i.bg_edge is True:
                temp = i.fill_patch(temp, color=(255, 0, 0))
            elif i.frame_edge is True:
                temp = i.fill_patch(temp, color=(0, 0, 255))
        show_image(temp, scale=2)
    def parse_frame(self):
        frame = self.raw_frame
        mask = np.ones(frame.shape[:-1])
        size = frame.shape[0] * frame.shape[1]
        self.patches = []
        self.bounding_boxes = []

        patches = []
        palette = {}
        patch_colors = []
        for i, row in enumerate(mask):
            for j, pix in enumerate(row):
                palette[tuple(frame[i][j])] = True
                if pix == 1:
                    mask[i][j] = 0
                    if self.is_background(i, j):
                        continue
                    node = Node(frame, i, j, mask=mask, indirect=self.indirect)

                    for x, y in node.coord_list():
                        mask[x][y] = 0

                    self.patches.append(node)
                    self.bounding_boxes.append(node.bounding_box)
                    patch_colors.append(tuple(frame[i][j]))

        self.palette = tuple(palette.keys())
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

    def isolate_offset_subgraphs(self):
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

        return (subgraphs, o_subgraphs)

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

    def fill_subgraph(self, subgraph, frame=None):
        if frame is None:
            frame = self.frame.raw_frame.copy()
        for i in subgraph:
            frame = i.fill_patch(frame)

        return frame

    def show(self, scale=None):
        show_image(self.raw_frame, scale=scale)

    def show_subgraph(self, subgraph):
        xs = []
        ys = []
        bbs = []
        for i in subgraph:
            xs.append(i.bounding_box[0][0])
            xs.append(i.bounding_box[1][0])
            ys.append(i.bounding_box[0][1])
            ys.append(i.bounding_box[1][1])
            bbs.append(i.bounding_box)
        bb = ((min(xs), min(ys)), (max(xs), max(ys)))

        left_most = list(filter(lambda p: p.bounding_box[0][1] == bb[0][1], subgraph))
        top_left_patch = list(sorted(left_most, key=lambda p: p.bounding_box[0][0]))[0]
        tlpbb = top_left_patch.bounding_box
        xt, yt = (tlpbb[0][0] - bb[0][0], tlpbb[0][1] - bb[0][1])

        shape = bb[1][0] - bb[0][0], bb[1][1] - bb[0][1], 3
        frame = np.zeros(shape, dtype='uint8')

        for x, y in top_left_patch.patch._patch.translate(xt, yt):
            frame[x][y] = top_left_patch.color

        for i in subgraph:
            ibb = i.bounding_box
            xt, yt = (ibb[0][0] - bb[0][0], ibb[0][1] - bb[0][1])
            for x, y in i.patch._patch.translate(xt, yt):
                frame[x][y] = i.color

        show_image(frame, scale=3)
