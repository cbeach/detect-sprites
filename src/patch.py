import json

import cv2
import numpy as np


class Frame:
    def __init__(self, frame):
        self.raw_frame = frame
        self.patches = self.get_patch_list(self.raw_frame)
        self.palette = set([i.color for i in self.patches])

    def get_patch_list(self, frame):
        mask = np.ones(frame.shape[:-1])
        bg_color = frame[0][0].copy()
        size = frame.shape[0] * frame.shape[1]
        patches = []
        for i, row in enumerate(mask):
            for j, pix in enumerate(row):
                if pix == 1:
                    if np.array_equal(bg_color, frame[i][j]) is True:
                        mask[i][j] = 0
                        continue
                    patches.append(Patch(frame, i, j, mask=mask))
        return patches

    def show(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(0)


class Patch:
    def __init__(self, frame, x_seed, y_seed, mask=None):
        self.color = tuple(frame[x_seed][y_seed])
        self.patch_as_list = self.get_patch_as_list(frame, mask, x_seed, y_seed)
        self.bounding_box = self.get_bounding_box(self.patch_as_list)
        self.patch_as_array = self.patch_list_to_array(self.patch_as_list, self.bounding_box)

    def get_patch_as_list(self, frame, mask, x, y):
        if mask is None:
            mask = np.ones(frame.shape[:-1])

        stack = [(x, y)]
        mask[x][y] = 0
        patch = []
        while len(stack) > 0:
            cur = stack.pop()
            patch.append(cur)
            for i, j in self.get_neighbors(frame, *cur):
                if mask[i][j] == 1:
                    stack.append((i, j))
                    mask[i][j] = 0
        return patch

    def get_neighbors(self, frame, x, y):
        neighbors = []
        if x > 0 and y > 0 and np.array_equal(frame[x-1][y-1], frame[x][y]):
            neighbors.append((x-1, y-1))
        if y > 0 and np.array_equal(frame[x][y-1], frame[x][y]):
            neighbors.append((x, y-1))
        if x < frame.shape[0] - 1 and y > 0 and np.array_equal(frame[x+1][y-1], frame[x][y]):
            neighbors.append((x+1, y-1))

        if x > 0 and np.array_equal(frame[x-1][y], frame[x][y]):
            neighbors.append((x-1, y))
        #if x > 0 and np.array_equal(frame[x][y], frame[x][y]):
        #    neighbors.append((x, y))
        if x < frame.shape[0] - 1 and np.array_equal(frame[x+1][y], frame[x][y]):
            neighbors.append((x+1, y))

        if x > 0 and y < frame.shape[1] - 1 and np.array_equal(frame[x-1][y+1], frame[x][y]):
            neighbors.append((x-1, y+1))
        if y < frame.shape[1] - 1 and np.array_equal(frame[x][y+1], frame[x][y]):
            neighbors.append((x, y+1))
        if x < frame.shape[0] - 1 and y < frame.shape[1] - 1 and np.array_equal(frame[x+1][y+1], frame[x][y]):
            neighbors.append((x+1, y+1))

        return neighbors

    def get_bounding_box(self, patch):
        x = [i[0] for i in patch]
        y = [i[1] for i in patch]
        return ((min(x), min(y)), (max(x) + 1, max(y) + 1))

    def patch_list_to_array(self, patch_list, bb):
        bb_size = (bb[1][0] - bb[0][0],  bb[1][1] - bb[0][1])
        min_x = bb[0][0]
        min_y = bb[0][1]

        p_arr = np.zeros(bb_size)
        for x, y in patch_list:
            p_arr[x - min_x][y - min_y] = 1
        return p_arr

    # Debugging --------------------
    def draw_bounding_box(self, frame):
        cp = frame.copy()
        cv2.rectangle(cp,
                      (self.bounding_box[0][1], self.bounding_box[0][0]),
                      (self.bounding_box[1][1], self.bounding_box[1][0]),
                      (0,255,0), 1)
        return cp

    def fill_patch(self, frame, patch):
        cp = frame.copy()
        for x, y in self.patch_as_list[patch]:
            cp[x][y][0] = 0
            cp[x][y][1] = 255
            cp[x][y][2] = 0
        return cp

    def __hash__(self):
        patch_hash = 0
        count = 0
        for i, row in enumerate(self.patch_as_array):
            for j, pix in enumerate(row):
                patch_hash += pix
                patch_hash << 1
        return patch_hash



if __name__ == "__main__":
    img = cv2.imread('1008-693.png')
    f = Frame(img)
    print(len(f.patches))
