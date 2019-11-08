from collections import defaultdict

import cv2
import numpy as np

from patch import Patch
from sprite_util import show_image
#from patch_store import PatchDB
from patch_graph import PatchGraph

class Frame:
    def __init__(self, frame, bg_color=None):
        self.raw_frame = frame

        if bg_color is None:
            self.bg_color = self.raw_frame[0][0].copy()
        else:
            self.bg_color = bg_color
        #self.db = PatchDB('patch_store.json')
        self.patch_index = defaultdict(list)
        self.pix_index = {}
        self.patches = self.patch_list()
        self.index_patches()
        self.palette = set([i.color for i in self.patches])

        #print(f'len patch_index: {len(self.patch_index)}')
        #print(f'len patches: {len(self.patches)}')

    def get_patches_by_coord(self, coords):
        try:
            iter(coords)
        except TypeError:
            return [self.pix_index[coords]]

        background_filtered_out = [c for c in coords if not self.is_background(c[0], c[1])]
        return [self.pix_index[c] for c in background_filtered_out]

    def index_patches(self):
        for patch in self.patches:
            self.patch_index[hash(patch)].append(patch)

            for coord in patch.patch_as_list:
                self.pix_index[coord] = patch

    def patch_list(self):
        frame = self.raw_frame
        mask = np.ones(frame.shape[:-1])
        size = frame.shape[0] * frame.shape[1]
        patches = []
        marked = self.raw_frame.copy()
        for i, row in enumerate(mask):
            for j, pix in enumerate(row):
                if pix == 1:
                    mask[i][j] = 0
                    if self.is_background(i, j):
                        continue
                    patch = Patch(frame, i, j, mask=mask)
                    for x, y in patch.patch_as_list:
                        mask[x][y] = 0
                    patches.append(patch)
                    #marked = patch.fill_patch(marked)
                    #show_image(marked)
                    #marked = patch.fill_patch(marked, color=(0, 0, 255))
                    #if i > 1:
                    #    sys.exit(0)

        return patches

    def is_background(self, x, y):
        return np.array_equal(self.bg_color, self.raw_frame[x][y]) is True

    def show(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(0)


if __name__ == "__main__":
    #img = cv2.imread('test-frame.png')
    #img = cv2.imread('1008-693.png')
    #f = Frame(img)
    #PatchGraph(f)

    r = np.array([0, 0, 255], dtype=np.uint8)
    g = np.array([0, 255, 0], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    c = np.array([255, 255, 0], dtype=np.uint8)
    m = np.array([255, 0, 255], dtype=np.uint8)
    w = np.array([255, 255, 255], dtype=np.uint8)
    a = np.array([128, 128, 128], dtype=np.uint8)

    ti1 = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
    ])

    tf1 = Frame(ti1, bg_color=np.array([0,0,0]))
    tg1 = PatchGraph(tf1)

    ti2 = np.array([
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
    ])
    tf2 = Frame(ti2, bg_color=np.array([0,0,0]))
    tg2 = PatchGraph(tf2)

    ti3 = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, w, w, w, w, c, c, c],
        [m, m, m, w, a, a, w, m, m, m],
        [r, r, r, w, a, a, w, r, r, r],
        [g, g, g, w, w, w, w, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
    ])
    tf3 = Frame(ti3, bg_color=np.array([0,0,0]))
    tg3 = PatchGraph(tf3)


