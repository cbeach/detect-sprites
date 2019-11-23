from collections import defaultdict

import cv2
import numpy as np

from patch import Patch, NormalizedPatch
from sprite_util import show_image, get_frame
from data_store import PatchDB

class Frame:
    @staticmethod
    def from_raw_frame(game, play_number, frame_number):
        return Frame(get_frame(game, play_number, frame_number))

    def __init__(self, frame, game='SuperMarioBros-Nes', bg_color=None):
        self.raw_frame = frame

        if bg_color is None:
            self.bg_color = self.raw_frame[0][0].copy()
        else:
            self.bg_color = bg_color

        #self.db = PatchDB('db/')

        self.patch_colors = []
        self.bounding_boxes = []
        self.patch_list()
        self.patch_index = defaultdict(list)
        self.pix_index = {}
        self.index_patches()

    def patch_list(self):
        frame = self.raw_frame
        mask = np.ones(frame.shape[:-1])
        size = frame.shape[0] * frame.shape[1]
        self.patches = []
        self.bounding_boxes = []

        marked = self.raw_frame.copy()
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
                    patch = Patch(frame, i, j, mask=mask)

                    for x, y in patch.patch_as_list:
                        mask[x][y] = 0

                    self.patches.append(patch)
                    self.bounding_boxes.append(patch.bounding_box)
                    patch_colors.append(tuple(frame[i][j]))

        #p_keys = []
        #for i in self.patches:
        #    norm_patch = NormalizedPatch(i)
        #    p_keys.append(self.db.add_patch(norm_patch))

        #self.p_keys = {k:v for k, v in zip(p_keys, self.patches)}
        self.palette = tuple(palette.keys())
        self.patch_colors = [self.color_as_palette(i) for i in patch_colors]

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

            for coord in patch.patch_as_list:
                self.pix_index[coord] = patch

    def is_background(self, x, y):
        return np.array_equal(self.bg_color, self.raw_frame[x][y]) is True

    def show(self, scale=None):
        show_image(self.raw_frame, scale=scale)


