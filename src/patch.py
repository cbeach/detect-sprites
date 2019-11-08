from collections import defaultdict
import itertools
import json
import math

import cv2
import numpy as np
from sprite_util import neighboring_points


class Patch:
    def __init__(self, frame, x_seed, y_seed, mask=None):
        self._is_self = defaultdict(lambda: False)
        self.color = tuple(frame[x_seed][y_seed])
        self.patch_as_list = self.get_patch_as_list(frame, mask, x_seed, y_seed)
        for coord in self.patch_as_list:
            self._is_self[coord] = True

        self.bounding_box = self.get_bounding_box()
        self.patch_as_array = self.patch_list_to_array(self.patch_as_list, self.bounding_box)
        self.my_hash = None
        self.my_hash = hash(self)
        self.my_hash_with_offset = None
        self.my_hash_with_offset = self.hash_with_offset()

    def is_self(self, coord):
        return self._is_self[coord]

    def get_patch_as_list(self, frame, mask, x, y):
        if mask is None:
            mask = np.ones(frame.shape[:-1])

        stack = [(x, y)]
        mask[x][y] = 0
        patch = []
        while len(stack) > 0:
            current_pixel = stack.pop()
            patch.append(current_pixel)
            for i, j in self.get_neighboring_self_pixels(frame, *current_pixel):
                if mask[i][j] == 1:
                    stack.append((i, j))
                    mask[i][j] = 0
        return patch

    def get_neighboring_self_pixels(self, frame, x, y):
        nbr_coords = neighboring_points(x, y, frame)

        self_pix = []
        for nx, ny in nbr_coords:
            if np.array_equal(frame[nx][ny], frame[x][y]):
                self_pix.append((nx, ny))
        return self_pix

    def get_neighboring_patch_pixels(self, frame_obj):
        nbr_pixels = []
        for x, y in self.patch_as_list:
            n = neighboring_points(x, y, frame_obj.raw_frame)
            nbr_pixels.extend(n)
        nbr_pixel_set = set(nbr_pixels)

        return [i for i in nbr_pixel_set if not self._is_self[i]]

    def get_bounding_box(self):
        x = [i[0] for i in self.patch_as_list]
        y = [i[1] for i in self.patch_as_list]
        return ((min(x), min(y)), (max(x) + 1, max(y) + 1))

    def patch_list_to_array(self, patch_list, bb):
        bb_size = (bb[1][0] - bb[0][0],  bb[1][1] - bb[0][1])
        min_x = bb[0][0]
        min_y = bb[0][1]

        p_arr = np.zeros(bb_size)
        for x, y in patch_list:
            p_arr[x - min_x][y - min_y] = 1
        return p_arr


    def __hash__(self):
        if self.my_hash is not None:
            return self.my_hash

        patch_hash = 1
        count = 0
        for i, pix in enumerate(self.patch_as_array.flatten()):
            patch_hash = patch_hash << 1
            if pix:
                patch_hash += 1

        self.my_hash = patch_hash
        return patch_hash

    def hash_with_offset(self):
        if self.my_hash_with_offset is not None:
            return self.my_hash_with_offset

        initial_hash = hash(self)
        top_left_corner = self.bounding_box[0]
        shifted_x = initial_hash << (len(bin(top_left_corner[0])) - 2)
        or_x = shifted_x | top_left_corner[0]
        shifted_y = or_x << (len(bin(top_left_corner[1])) - 2)
        with_offset = shifted_y | top_left_corner[1]

        self.my_hash_with_offset = with_offset
        return with_offset

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
        for x, y in self.patch_as_list:
            cp[x][y][0] = color[0]
            cp[x][y][1] = color[1]
            cp[x][y][2] = color[2]
        return cp

