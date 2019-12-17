from collections import defaultdict
import itertools
import json
import math
import sys

import cv2
import numpy as np
from sprite_util import neighboring_points
from db.models import NodeM, PatchM


class Node:
    def __init__(self, frame, x_seed, y_seed, mask=None, indirect=True):
        self.patch = Patch(frame, x_seed, y_seed, mask, indirect)
        self.color = frame[x_seed][y_seed]
        self._coord_list = None
        self.bounding_box = self.patch._bb

        self.my_offset_hash = None
        self.my_offset_hash = self.offset_hash()

        self._self_pix = {}
        for coord in self.coord_list():
            self._self_pix[coord] = True

        self.frame_edge = False
        self.bg_edge = False

    def mark_as_frame_edge(self):
        self.frame_edge = True

    def mark_as_bg_edge(self):
        self.bg_edge = True

    def coord_list(self):
        if self._coord_list:
            return self._coord_list
        else:
            return self.patch.translate(self.bounding_box[0][0], self.bounding_box[0][1])

    def is_self(self, coord):
        return self._self_pix.get(coord, False)

    def get_neighboring_self_pixels(self, frame, x, y):
        nbr_coords = neighboring_points(x, y, frame, self.indirect_neighbors)

        self_pix = []
        for nx, ny in nbr_coords:
            if np.array_equal(frame[nx][ny], frame[x][y]):
                self_pix.append((nx, ny))
        return self_pix

    def get_neighboring_patch_pixels(self, frame):
        nbr_pixels = []
        for x, y in self.coord_list():
            n = neighboring_points(x, y, frame)
            nbr_pixels.extend(n)
        nbr_pixel_set = set(nbr_pixels)

        return [i for i in nbr_pixel_set if not self.is_self(i)]

    def __hash__(self):
        return hash(self.patch)

    def offset_hash(self):
        if self.my_offset_hash is not None:
            return self.my_offset_hash

        initial_hash = hash(self)
        top_left_corner = self.bounding_box[0]
        shifted_x = initial_hash << (len(bin(top_left_corner[0])) - 2)
        or_x = shifted_x | top_left_corner[0]
        shifted_y = or_x << (len(bin(top_left_corner[1])) - 2)
        with_offset = shifted_y | top_left_corner[1]

        self.my_offset_hash = with_offset
        return with_offset

    def get_relative_offset(self, other_patch):
        other_bb = other_patch.bounding_box
        return ((other_bb[0][0] - self.bounding_box[0][0], other_bb[0][1] - self.bounding_box[0][1]),
                (other_bb[1][0] - self.bounding_box[1][0], other_bb[1][1] - self.bounding_box[1][1]))

    # Debugging --------------------
    def draw_bounding_box(self, frame):
        cp = frame.copy()
        cv2.rectangle(cp,
                      (self.bounding_box[0][1], self.bounding_box[0][0]),
                      (self.bounding_box[1][1], self.bounding_box[1][0]),
                      (0,255,0), 1)
        return cp

    def fill_patch(self, frame, color=(0, 255, 0)):
        cp = frame.copy()
        for x, y in self.coord_list():
            cp[x][y][0] = color[0]
            cp[x][y][1] = color[1]
            cp[x][y][2] = color[2]
        return cp


class Patch:
    __PATCHES = {}
    def __init__(self, frame, x_seed, y_seed, mask=None, indirect=True, ds=None):
        # Pull the patches out of the db
        if len(Patch.__Patches) == 0 and ds is not None:
            ds.session

        temp_patch = Patch._sub_patch(frame, x_seed, y_seed, mask, indirect)
        self._bb = temp_patch._bb

        if Patch.__PATCHES.get(hash(temp_patch), None) is None:
            self._patch = Patch.__PATCHES[hash(temp_patch)] = temp_patch
        else:
            self._patch = Patch.__PATCHES[hash(temp_patch)]

    def shape(self):
        return self._patch.shape()

    def area(self):
        return self._patch.area()

    def translate(self, x, y):
        return self._patch.translate(x, y)

    def __hash__(self):
        return hash(self._patch)

    #def __getstate__(self):
    #    return {
    #        '__PATCHES': Patch.__PATCHES,
    #        '_patch': self._patch,
    #        '_bb': self._bb,
    #    }

    #def __setstate__(self, data):
    #    Patch.__PATCHES = data['__PATCHES']
    #    self._patch = data['_patch']
    #    self._bb = data['_bb']

class _patch:
    def __init__(self, frame, x_seed, y_seed, mask=None, indirect=True, model=None):
        if model is None:
            self.indirect_neighbors = indirect
            self.coord_list, self._bb = self._get_coord_list(frame, mask, x_seed, y_seed)
            self._patch = self._coord_list_to_array()
            self.my_hash = None
            self.my_hash = hash(self)
        else:
            self._from_model(model)

    def _from_model(self, model):


    def _get_coord_list(self, frame, mask, x, y):
        if mask is None:
            mask = np.ones(frame.shape[:-1])

        stack = [(x, y)]
        mask[x][y] = 0
        patch = []
        while len(stack) > 0:
            current_pixel = stack.pop()
            patch.append(current_pixel)
            nbr_coords = neighboring_points(current_pixel[0], current_pixel[1], frame, self.indirect_neighbors)

            for i, j in nbr_coords:
                if mask[i][j] == 1 and np.array_equal(frame[i][j], frame[x][y]):
                    stack.append((i, j))
                    mask[i][j] = 0

        # bounding_box
        x = [i[0] for i in patch]
        y = [i[1] for i in patch]
        bb = ((min(x), min(y)), (max(x) + 1, max(y) + 1))

        # normalize
        norm_patch = [(i[0] - bb[0][0], i[1] - bb[0][1]) for i in patch]

        return norm_patch, bb

    def _coord_list_to_array(self):
        x = [i[0] for i in self.coord_list]
        y = [i[1] for i in self.coord_list]
        bb = ((min(x), min(y)), (max(x) + 1, max(y) + 1))

        size = (bb[1][0] - bb[0][0],  bb[1][1] - bb[0][1])
        min_x = bb[0][0]
        min_y = bb[0][1]

        p_arr = np.zeros(size)
        for x, y in self.coord_list:
            p_arr[x - min_x][y - min_y] = 1
        return p_arr

    def shape(self):
        return tuple(self._patch.shape)

    def area(self):
        return len(self.coord_list)

    def translate(self, x, y):
        return [(i[0] + x, i[1] + y) for i in self.coord_list]

    def __hash__(self):
        if self.my_hash is not None:
            return self.my_hash

        patch_hash = 1
        for i, pix in enumerate(self._patch.flatten()):
            patch_hash = patch_hash << 1
            if pix:
                patch_hash += 1

        x, y = bytes(self._patch.shape)
        shaped_hash = patch_hash
        shifted_x = shaped_hash << 8
        or_x = shifted_x | x
        shifted_y = or_x << 8
        with_shape = shifted_y | y

        self.my_hash = with_shape
        return with_shape
