from collections import defaultdict
import itertools
import json
import math
import sys

import cv2
import numpy as np

from .sprite_util import flattened_neighboring_points, neighboring_points, conjugate_numbers
from .db.models import EdgeM, NodeM, PatchM
from .db.data_store import DataStore
from .db.data_types import BoundingBox, Color, FrameID, Mask, encode_frame_id

class FrameEdge:
    def is_background(self):
        return False

    def is_frame_edge(self):
        return True

    def is_special(self):
        return True

class Background:
    def is_background(self):
        return True

    def is_frame_edge(self):
        return False

    def is_special(self):
        return True


frame_edge_node = FrameEdge()
background_node = Background()

frame_edge_nodes = defaultdict(list)
background_nodes = defaultdict(list)

class Node:
    def __init__(self, frame, x_seed, y_seed, play_number, frame_number, mask=None, indirect=True, ds=None):
        self.play_number = play_number
        self.frame_number = frame_number
        self.patch = Patch(frame, x_seed, y_seed, mask, indirect, ds=ds)
        self.color = frame[x_seed][y_seed]
        self._coord_list = None
        self.bounding_box = self.patch._bb
        self.indirect = indirect
        self.patch_hash = hash(self.patch)

        self.hash_memoization = {}
        self.hash_memoization['offset'] = self._binary_offset()
        self.hash_memoization['color'] = self._binary_color()

        self.my_offset_hash = None
        self.my_offset_hash = self.offset_hash()

        self._self_pix = {}
        for coord in self.coord_list():
            self._self_pix[coord] = True

        self.frame_edge = False
        self.bg_edge = False
        self.neighbors = []
        self.my_neighborhood_hash = None

    def mark_as_frame_edge(self):
        self.frame_edge = True
        self.neighbors.append(frame_edge_node)
        frame_edge_nodes[(self.play_number, self.frame_number)].append(self)

    def touches_edge(self):
        return self.frame_edge

    def touches_bg(self):
        return self.bg_edge

    def mark_as_bg_edge(self):
        self.bg_edge = True
        self.neighbors.append(background_node)
        background_nodes[(self.play_number, self.frame_number)].append(self)

    def coord_list(self):
        if self._coord_list:
            return self._coord_list
        else:
            return self.patch.translate(self.bounding_box[0][0], self.bounding_box[0][1])

    def is_self(self, coord):
        return self._self_pix.get(coord, False)

    def is_background(self):
        return False

    def is_frame_edge(self):
        return False

    def is_special(self):
        return False

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
            n = neighboring_points(x, y, frame, indirect=self.indirect)
            nbr_pixels.extend(n)
        nbr_pixel_set = set(nbr_pixels)

        return [i for i in nbr_pixel_set if not self.is_self(i)]

    def __hash__(self):
        return self.patch_hash

    def _binary_offset(self, relative=None):
        if relative is None:
            if 'offset' in self.hash_memoization:
                return self.hash_memoization['offset']
            shifted_x = conjugate_numbers(self.bounding_box[0][0])
            shifted_y = conjugate_numbers(self.bounding_box[0][1], seed=shifted_x)
            self.hash_memoization['offset'] = shifted_y
        else:
            x, y = self.get_relative_offset(relative)
            other = relative.master_hash(color=True)
            shifted_x = conjugate_numbers(x, seed=other)
            shifted_y = conjugate_numbers(y, seed=shifted_x)
        return shifted_y

    def _binary_color(self):
        if 'color' in self.hash_memoization:
            return self.hash_memoization['color']

        r = conjugate_numbers(self.color[2])
        rg = conjugate_numbers(self.color[1], seed=r)
        rgb = conjugate_numbers(self.color[0], seed=rg)
        self.hash_memoization['color'] = rgb
        return rgb

    def offset_hash(self):
        return self.master_hash(offset=True)

    def color_hash(self):
        return self.master_hash(color=True)

    def ch(self):
        return self.color_hash()

    def oh(self):
        return self.offset_hash()

    def coh(self):
        return self.color_offset_hash()

    def color_offset_hash(self):
        return self.master_hash(offset=True, color=True, relative=None)

    def master_hash(self, offset=False, color=False, relative=None):
        if relative is not None:
            return conjugate_numbers(
                        self._binary_color(),
                        conjugate_numbers(self._binary_offset(relative=relative), seed=hash(self))
                    )
        elif offset is True and color is True:
            if self.hash_memoization.get((True, True), None) is None:
                self.hash_memoization[(True, True)] = conjugate_numbers(
                    self._binary_color(),
                    conjugate_numbers(self._binary_offset(), seed=hash(self))
                )
        elif offset is True and color is False:
            if self.hash_memoization.get((True, False), None) is None:
                self.hash_memoization[(offset, color)] = conjugate_numbers(self._binary_offset(relative), seed=hash(self))
        elif offset is False and color is True:
            if self.hash_memoization.get((False, True), None) is None:
                self.hash_memoization[(offset, color)] = conjugate_numbers(self._binary_color(), seed=hash(self))
        elif offset is False and color is False:
            return hash(self)


        return self.hash_memoization[(offset, color)]

    def neighborhood_hash(self):
        if self.my_neighborhood_hash is not None:
            return self.my_neighborhood_hash

        initial_hash = hash(self)
        for nbr in self.neighbors:
            nbr_hash = hash(nbr)
            shifted = initial_hash << (len(bin(nbr_hash)) - 2)
            initial_hash = initial_hash | nbr_hash

        self.my_neighborhood_hash = initial_hash
        return self.my_neighborhood_hash

    def get_relative_offset(self, other_patch):
        other_bb = other_patch.bounding_box
        return (other_bb[0][0] - self.bounding_box[0][0], other_bb[0][1] - self.bounding_box[0][1])

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

    def edge_hashes(self):
        e_hashes = []
        for right in self.neighbors:
            if right is background_node or right is frame_edge_node:
                continue
            else:
                print('normal_patch')
                bb1, bb2 = self.bounding_box, right.bounding_box
                x_o, y_o = bb1[0][0] - bb2[0][0], bb1[0][1] - bb2[0][1]
                offset = (x_o, y_o)

            full_offset = conjugate_numbers(offset[1], seed=offset[0], num_length=16)
            temp = conjugate_numbers(full_offset, num_length=32, seed=right.master_hash(color=True, offset=True))
            print(temp)
            e_hashes.append(temp)

        return self.master_hash(color=True, offset=True), e_hashes

    def get_mask(self):
        return self.patch.get_mask()

    def neighborhood_similarity(self, sprite_node):
        if self != sprite_node:
            raise ValueError('Nodes must be equal to compare neighborhoods')
        chs, chspr = self.master_hash(color=True), sprite_node.master_hash(color=True),
        sns, sprns = (set([(self.get_relative_offset(n), n) for n in self.neighbors if not n.is_background() and not n.is_frame_edge()]),
            set([(sprite_node.get_relative_offset(n), n) for n in sprite_node.neighbors if not n.is_background() and not n.is_frame_edge()]))
        return [i[1] for i in sns - sprns]

    def neighborhood_diff(self, sprite_node):
        if self != sprite_node:
            raise ValueError('Nodes must be equal to compair neighborhoods')
        chs, chspr = self.master_hash(color=True), sprite_node.master_hash(color=True),
        sns, sprns = (set([(self.get_relative_offset(n), n) for n in self.neighbors if not n.is_background() and not n.is_frame_edge()]),
            set([(sprite_node.get_relative_offset(n), n) for n in sprite_node.neighbors if not n.is_background() and not n.is_frame_edge()]))
        #sns, sprns = set(self.neighbors), set(sprite_node.neighbors)
        return self == sprite_node and sprns.issubset(sns)

    @classmethod
    def set_comparison_context(cls, color=True, offset=False):
        cls.cmp_ctx = {'color': color, 'offset': offset}

    @classmethod
    def get_comparison_context(cls):
        return cls.cmp_ctx

    def __eq__(self, other):
        if self.is_background() or self.is_frame_edge() or other.is_background() or other.is_frame_edge():
            return False
        else:
            ctx = Node.get_comparison_context()
            return self.master_hash(**Node.get_comparison_context()) == other.master_hash(**Node.get_comparison_context())

    def __lt__(self, other):
        if self.is_background() or self.is_frame_edge() or other.is_background() or other.is_frame_edge():
            return False
        else:
            return self.master_hash(**Node.get_comparison_context()) < other.master_hash(**Node.get_comparison_context())

    def __le__(self, other):
        if self.is_background() or self.is_frame_edge() or other.is_background() or other.is_frame_edge():
            return False
        else:
            return self.master_hash(**Node.get_comparison_context()) <= other.master_hash(**Node.get_comparison_context())

    def __ne__(self, other):
        if self.is_background() or self.is_frame_edge() or other.is_background() or other.is_frame_edge():
            return False
        else:
            return self.master_hash(**Node.get_comparison_context()) != other.master_hash(**Node.get_comparison_context())

    def __gt__(self, other):
        if self.is_background() or self.is_frame_edge() or other.is_background() or other.is_frame_edge():
            return False
        else:
            return self.master_hash(**Node.get_comparison_context()) > other.master_hash(**Node.get_comparison_context())

    def __ge__(self, other):
        if self.is_background() or self.is_frame_edge() or other.is_background() or other.is_frame_edge():
            return False
        else:
            return self.master_hash(**Node.get_comparison_context()) >= other.master_hash(**Node.get_comparison_context())

    def cull_neighbors(self):
        mhashes = {p.master_hash(color=True, offset=True):p for p in self.neighbors}
        self.neighbors = list(mhashes.values())

    def get_neighbors(self):
        return [n for n in self.neighbors if not n.is_special()]

    def area(self):
        return self.patch.area()

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
        # TODO: I should re-write the above, but I'm in the middle of debugging
        # in jupyter. I'll save the below snippent for later

        # cp = frame.copy()
        # color = np.array(color, dtype='uint8')
        # for x, y in self.coord_list():
        #     cp[x][y] = color
        # return cp

    def fill_neighborhood(self, frame):
        frame = self.fill_patch(frame)
        grad_step = math.floor(256 / len(self.neighbors) + 1)
        for i, p in enumerate(self.neighbors):
            if not p.is_frame_edge() and not p.is_background():
                frame = p.fill_patch(frame, color=(255, grad_step * (i + 1), grad_step * (i + 1)))
        return frame


class Patch:
    _PATCHES = {}
    @staticmethod
    def init_patch_db(ds=None):
        pass
    #    # Pull the patches out of the db
    #    if ds is None:
    #        ds = ds = DataStore('temp.db', games_path='./games.json')

    #    if len(Patch._PATCHES) == 0 and ds is not None:
    #        db_sess = ds.Session()
    #        db_sess.commit()
    #        if db_sess.query(PatchM).count() > 0:
    #            # Load patches from db
    #            for i in db_sess.query(PatchM).all():
    #                p = _patch(model=i)
    #                Patch._PATCHES[hash(p)] = p

    def __init__(self, frame, x_seed, y_seed, mask=None, indirect=True, ds=None):
        self.indirect = indirect
        temp_patch = _patch(frame, x_seed, y_seed, mask, indirect)
        self._bb = temp_patch._bb

        self._patch = Patch._PATCHES[hash(temp_patch)] = temp_patch
        #if Patch._PATCHES.get(hash(temp_patch), None) is None:
        #    self._patch = Patch._PATCHES[hash(temp_patch)] = temp_patch
        #else:
        #    self._patch = Patch._PATCHES[hash(temp_patch)]

        self.patch_hash = hash(self._patch)

    def shape(self):
        return self._patch.shape()

    def area(self):
        return self._patch.area()

    def coord_list(self):
        return self._patch.coord_list

    def translate(self, x, y):
        return self._patch.translate(x, y)

    def __hash__(self):
        return self.patch_hash

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

    def get_mask(self):
        return self._patch._patch


class _patch:
    def __init__(self, frame=None, x_seed=None, y_seed=None, mask=None, indirect=True, model=None):
        if model is None:
            self.indirect_neighbors = indirect
            self.coord_list, self._bb = self._get_coord_list(frame, mask, x_seed, y_seed)
            self._patch = self._coord_list_to_array()
        else:
            self._from_model(model)

        self.my_hash = self.patch_hash()

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
        return self.my_hash

    def patch_hash(self):
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

        return with_shape

    def store(self, sess):
        if sess.query(PatchM.id).filter(PatchM.id==hash(self)).scalar() is None:
            sess.add(PatchM(id=hash(self), shape=self._patch.shape, indirect=self.indirect_neighbors, mask=self._patch))

    def __eq__(self, other):
        return hash(self) == hash(other)


Node.set_comparison_context()

