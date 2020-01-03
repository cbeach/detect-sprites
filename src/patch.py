from collections import defaultdict
import itertools
import json
import math
import sys

import cv2
import numpy as np
from sprite_util import neighboring_points
from db.models import EdgeM, NodeM, PatchM
from db.data_store import DataStore
from db.data_types import BoundingBox, Color, FrameID, Mask, encode_frame_id


class Node:
    def __init__(self, frame, x_seed, y_seed, mask=None, indirect=True, ds=None):
        #if ds is not None:
        #    self.ds = ds
        #else:
        #    self.ds = ds = DataStore('temp.db', games_path='./games.json')

        self.patch = Patch(frame, x_seed, y_seed, mask, indirect, ds=ds)
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
        self.neighbors = []

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

    def store(self, game, play_number, frame_number, commit=False, ds=None, sess=None):
        if ds is None:
            ds = self.ds

        if sess is None:
            sess = ds.Session()

        if sess.query(PatchM).filter(PatchM.id==hash(self.patch)).count() == 0:
            self.patch.store(ds)

        pM = sess.query(PatchM).filter(PatchM.id==hash(self.patch)).first()
        node = NodeM(
            patch=pM.id,
            color=self.color,
            bb=self.bounding_box,
            game_id=game.id,
            play_number=play_number,
            frame_number=frame_number,
        )

        if commit is True:
            sess.add(node)
            sess.commit()
        else:
            return node


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
    _PATCHES = {}
    @staticmethod
    def init_patch_db(ds=None):
        # Pull the patches out of the db
        if ds is None:
            ds = ds = DataStore('temp.db', games_path='./games.json')

        if len(Patch._PATCHES) == 0 and ds is not None:
            db_sess = ds.Session()
            db_sess.commit()
            if db_sess.query(PatchM).count() > 0:
                # Load patches from db
                for i in db_sess.query(PatchM).all():
                    p = _patch(model=i)
                    Patch._PATCHES[hash(p)] = p

    def __init__(self, frame, x_seed, y_seed, mask=None, indirect=True, ds=None):
        self.indirect = indirect
        temp_patch = _patch(frame, x_seed, y_seed, mask, indirect)
        self._bb = temp_patch._bb

        if Patch._PATCHES.get(hash(temp_patch), None) is None:
            self._patch = Patch._PATCHES[hash(temp_patch)] = temp_patch
        else:
            self._patch = Patch._PATCHES[hash(temp_patch)]

    def shape(self):
        return self._patch.shape()

    def area(self):
        return self._patch.area()

    def translate(self, x, y):
        return self._patch.translate(x, y)

    def __hash__(self):
        return hash(self._patch)

    def store(self, ds=None):
        sess = ds.Session()
        for i in self._PATCHES.values():
            i.store(sess)
        sess.commit()

    def _load_all(self):
        sess = self.ds.Session()
        for i in sess.query(PatchM).all():
            p = _patch(model=i)
            self._PATCHES[hash(p)] = p


class _patch:
    def __init__(self, frame=None, x_seed=None, y_seed=None, mask=None, indirect=True, model=None):
        if model is None:
            self.indirect_neighbors = indirect
            self.coord_list, self._bb = self._get_coord_list(frame, mask, x_seed, y_seed)
            self._patch = self._coord_list_to_array()
        else:
            self._from_model(model)

        self.my_hash = None
        self.my_hash = hash(self)

    def _from_model(self, model):
        self.indirect_neighbors = model.indirect
        self.coord_list = []
        for x, row in enumerate(model.mask):
            for y, p in enumerate(row):
                if p == True:
                    self.coord_list.append((x, y))
        self._patch = self._coord_list_to_array()

    def _get_coord_list(self, frame, mask, x, y):
        if mask is None:
            mask = np.zeros(frame.shape[:-1])

        stack = [(x, y)]
        mask[x][y] = 0
        patch = []
        while len(stack) > 0:
            current_pixel = stack.pop()
            patch.append(current_pixel)
            nbr_coords = neighboring_points(current_pixel[0], current_pixel[1], frame, self.indirect_neighbors)

            for i, j in nbr_coords:
                if mask[i][j] == False and np.array_equal(frame[i][j], frame[x][y]):
                    stack.append((i, j))
                    mask[i][j] = True

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

        p_arr = np.zeros(size, dtype=bool)
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

    def store(self, sess):
        if sess.query(PatchM.id).filter(PatchM.id==hash(self)).scalar() is None:
            sess.add(PatchM(id=hash(self), shape=self._patch.shape, indirect=self.indirect_neighbors, mask=self._patch))


def main():
    r = np.array([0, 0, 255], dtype=np.uint8)
    g = np.array([0, 255, 0], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    c = np.array([255, 255, 0], dtype=np.uint8)
    m = np.array([255, 0, 255], dtype=np.uint8)

    w = np.array([255, 255, 255], dtype=np.uint8)
    a = np.array([128, 128, 128], dtype=np.uint8)

    ti1 = np.array([
        [r, r, r, r, r, g, g, g, g, g],
        [r, r, r, r, r, g, g, g, g, g],
        [r, r, r, r, r, g, g, g, g, g],
        [r, r, r, r, r, g, g, g, g, g],
        [r, r, r, r, r, c, g, g, g, g],
        [b, b, b, b, r, c, c, c, c, c],
        [b, b, b, b, b, c, c, c, c, c],
        [b, b, b, b, b, c, c, c, c, c],
        [b, b, b, b, b, c, c, c, c, c],
        [b, b, b, b, b, c, c, c, c, c],
    ], dtype='uint8')


    patches = {}
    patches['r'] = Patch(ti1, 0, 0, mask=np.zeros(ti1.shape[:-1]))
    patches['g'] = Patch(ti1, 0, 5, mask=np.zeros(ti1.shape[:-1]))
    patches['b'] = Patch(ti1, 5, 0, mask=np.zeros(ti1.shape[:-1]))
    patches['c'] = Patch(ti1, 5, 5, mask=np.zeros(ti1.shape[:-1]))
    return patches

if __name__ == '__main__':
    ds = DataStore('temp.db', games_path='./games.json')
    r = np.array([0, 0, 255], dtype=np.uint8)
    ti1 = np.array([
        [r, r],
        [r, r],
    ], dtype='uint8')
    j = main()
    p = Patch(ti1, 0, 0, mask=np.zeros(ti1.shape[:-1]))
    p.store()
    sess = p.ds.Session()
    for i in sess.query(PatchM).all():
        print(i)
    print(len(p._PATCHES), p._PATCHES)

