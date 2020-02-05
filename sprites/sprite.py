from collections import defaultdict
import sys

import numpy as np
import cv2

from .patch import Node, Patch
from .sprite_util import show_image, show_images

ig = cv2.imread('./sprites/test/images/ground.png')


def matching_patches(node, graph):
    return [n for n in graph.patches if hash(n) == hash(node)]


def fit_bounding_box(img, bb):
    l, r = bb
    lx, ly = l
    rx, ry = r
    bblx = 0 if lx < 0 else lx
    bbrx = img.shape[0] - 1 if rx >= img.shape[0] else rx
    bbly = 0 if ly < 0 else ly
    bbry = img.shape[1] - 1 if ry >= img.shape[1] else ry
    return bblx, bbly, bbrx, bbry


class Sprite:
    def __init__(self, graphlet=None, bg_color=None, palette=None, path=None):
        if graphlet is None and bg_color is None and palette is None:
            self.load(path)
        else:
            self._graphlet_init(graphlet, bg_color, palette)

    def _graphlet_init(self, graphlet, bg_color, palette):
        sorted_nodes = sorted([node for node in graphlet.nodes], key=lambda n: hash(n.patch))
        self.palette = np.array([bg_color] + [n.color for n in sorted_nodes], dtype='uint8')
        self.patches = [i.patch for i in sorted_nodes]
        self.hashes = [hash(p) for p in self.patches]
        self.sprite = graphlet.palettized.copy().astype('int32')
        ox, oy = graphlet.bb[0][0], graphlet.bb[0][1]

        for i, node in enumerate(sorted_nodes):
            for x, y in node.coord_list():
                self.sprite[x - ox][y - oy] = i
        self.bbs = [self._get_bounding_box(i) for i, _ in enumerate(self.patches)]

        self.adjacency_matrix = am = np.zeros([len(self.hashes)] * 2, dtype=bool)
        for i, row in enumerate(self.adjacency_matrix):
            left = sorted_nodes[i]
            for j, cell in enumerate(row):
                right = sorted_nodes[j]
                am[i][j] = any([nbr is right for nbr in left.get_neighbors()])

    def palette_from_image(self, img, bg_color, slice_params=(None, None, None, None)):
        palette = np.zeros((len(self.hashes) + 1, 3), dtype='uint8')
        palette[0] = bg_color
        a, b, c, d = slice_params
        sprite = self.sprite[a:b, c:d]
        show_image(img, scale=3.0)
        for x, row in enumerate(img):
            for y, pix in enumerate(row):
                palette[self.sprite[x][y] + 1] = pix
        return palette

    def to_image(self, palette=None, reference_img=None, bg_color=None, slice_params=(None, None, None, None)):
        a, b, c, d = slice_params
        if reference_img.shape[:2] != self.sprite[a:b, c:d].shape:
            raise ValueError(f'reference image must be the same shape as self.sprite[a:b, c:d]. {reference_img.shape} != {self.sprite[a:b, c:d].shape}')

        if palette is not None:
            palette = np.array([list(c) + [255] for c in palette], dtype='uint8')
            palette[0][3] = 0
        elif reference_img is not None:
            palette = self.palette_from_image(reference_img, bg_color)
            palette = np.hstack((palette, np.full((palette.shape[0], 1), 255)))
            p2 = np.hstack((self.palette, np.full((self.palette.shape[0], 1), 255)))
            palette[0][3] = 0
            p2[0][3] = 0
        #print('self.p', p2)
        #print('     p', palette)
        #print('self.p', p2.shape)
        #print('     p', palette.shape)
        #print('self.p', p2.dtype)
        #print('     p', palette.dtype)
        #print(np.array_equal(p2, palette))

        sx, sy = self.sprite.shape
        img = np.zeros((sx, sy, 4), dtype='uint8')
        sprite = (self.sprite.copy() + 1)[a:b, c:d]
        for x, row in enumerate(img):
            for y, pix in enumerate(row):
                img[x][y] = palette[sprite[x][y]]
        return img

    def __hash__(self):
        return int(f'{self.sprite.shape[0]}{self.sprite.shape[1]}{"".join([str(i) for i in self.sprite.flatten() + 1])}')

    def save(self, path):
        np.savez(path, hashes=self.hashes, sprite=self.sprite, adjacency_matrix=self.adjacency_matrix, palette=self.palette)

    def load(self, path):
        data = np.load(path)
        self.hashes = list(data['hashes'])
        self.patches = [Patch._PATCHES[i] for i in self.hashes]
        self.sprite = data['sprite']
        self.adjacency_matrix = data['adjacency_matrix']
        self.bbs = [self._get_bounding_box(i) for i, _ in enumerate(self.patches)]
        self.palette = data['palette']

    def __eq__(self, other):
        return np.array_equal(self.hashes, other.hashes) and np.array_equal(self.adjacency_matrix, other.adjacency_matrix) and np.array_equal(self.sprite, other.sprite)

    def best_pairs(self, pairs, graph):
        counts = defaultdict(int)
        for r, l in pairs:
            counts[hash(self.hashes[r])] += 1
        print(counts)
        p2 = [self.find_pairs_in_graph(p, graph) for p in pairs]
        l = [len(p) for p in p2]
        ind = l.index(max(l))
        return pairs[ind], p2[ind]

    def remove_from_frame(self, graph, img):
        root_pairs = self.find_feasible_root_pairs(graph)
        ref_pair, anchors = self.best_pairs(root_pairs, graph)
        img2 = anchors[0][0].fill_patch(graph.raw_frame)
        img2 = anchors[0][1].fill_patch(img2)
        show_image(img2, scale=3)
        coords = self.get_sprite_coords(ref_pair, anchors)
        return self.cull_sprites(graph, coords, img=img)

    def find_pairs_in_graph(self, pair, graph):
        ach, bch = hash(pair[0]), hash(pair[1])
        found = []
        for p in graph.patches:
            for n in p.get_neighbors():
                if self.pairs_equal(pair, (p, n)) is True:
                    found.append((p, n))
        return found

    def pairs_equal(self, aind, b):
        r0, r1 = aind[0], aind[1]
        p0, p1 = self.patches[r0], self.patches[r1]
        return (hash(p0) == hash(b[0]) and hash(p1) == hash(b[1]) and
                self.get_relative_offset(r0, r1) == b[0].get_relative_offset(b[1]))

    def _get_bounding_box(self, ind):
        x, y = np.where(self.sprite == ind)
        return ((min(x), min(y)), (max(x) + 1, max(y) + 1))

    def get_relative_offset(self, lind, rind):
        lbb, rbb = self.bbs[lind], self.bbs[rind]
        return (rbb[0][0] - lbb[0][0], rbb[0][1] - lbb[0][1])

    def find_feasible_root_pairs(self, frame):
        potential_root_pairs = []
        for i, root in enumerate(self.patches):
            potential_root_pairs.extend([(i, j) for j in np.nonzero(self.adjacency_matrix[i])[0]])
        print()
        print('potential_root_pairs', potential_root_pairs)
        root_pair_set = list(set([(rind, lind) for rind, lind in potential_root_pairs]))
        print()
        print('root_pair_set', root_pair_set)
        root_pairs = sorted(root_pair_set, key=lambda p: hash(self.patches[p[0]]), reverse=True)
        print()
        print('root_pairs', root_pairs)
        print()
        print(len(self.hashes), self.hashes)
        print(len(frame.hashes), frame.hashes)
        for i in self.hashes:
            if i in frame.hashes:
                print(i)
        sys.exit(0)
        feasible, candidates = [], [matching_patches(i, frame) for i in self.patches]
        for i in candidates:
            print('candidates', i)
        for i, p in enumerate(candidates):
            cand_pairs = []
            for j in p:
                for k in j.get_neighbors():
                    cand_pairs.append((j, k))

            root_pairs.reverse()
            for j, rpind in enumerate(root_pairs):
                for k, cp in enumerate(cand_pairs):
                    if self.pairs_equal(rpind, cp):
                        feasible.append(rpind)

        return list(set(feasible))

    def get_sprite_coords(self, ref, anchors):
        coords = []
        ox, oy = self.bbs[ref[0]][0]
        sx, sy = self.sprite.shape
        for a, b in anchors:
            aox, aoy = a.bounding_box[0]
            coords.append(((aox - ox,  aoy - oy), (aox - ox + sx, aoy - oy + sy)))
        return coords

    def cull_sprites(self, graph, coords, img=None):
        if img is None:
            img = graph.raw_frame.copy()

        for i, c in enumerate(coords):  # top left, bottom right corners
            print('coord', c)
            l, r = c
            lx, ly = l
            rx, ry = r
            sx, sy = self.sprite.shape

            bblx, bbly, bbrx, bbry = fit_bounding_box(img, (l, r))
            nlx, nly, nrx, nry = bblx - lx, bbly - ly, sx - (rx - bbrx), sy - (ry - bbry)
            print('slice_spec', bblx, bbrx, bbly, bbry)
            img_slice = img[bblx:bbrx, bbly:bbry, :]

            print('img')
            show_image(cv2.rectangle(img.copy(), (bbly, bblx), (bbry, bbrx), [255, 0, 0]), scale=3)
            print('img_slice')
            show_images((img_slice, ig), scale=3)
            s_img = self.to_image(reference_img=img_slice, bg_color=graph.bg_color, slice_params=(nlx, nrx, nly, nry))[:, :, :-1]
            print('s_img')
            show_image(s_img)
            sys.exit(0)

            #show_image(img_slice, scale=8.0)
            #show_image(s_img, scale=8.0)
            for x, row in enumerate(img[bblx:bbrx]):
                for y, pix in enumerate(row[bbly:bbry]):
                    if np.array_equal(pix, s_img[x][y]):
                        pix[0] = graph.bg_color[0]
                        pix[1] = graph.bg_color[1]
                        pix[2] = graph.bg_color[2]

        return img

    def get_patches_in_bounding_box(self, graph, bounding_box, partial=True):
        tlc, brc = bounding_box
        coord_map = np.zeros(graph.raw_frame.shape[:2], dtype='uint16')
        sx, sy = self.sprite.shape[:2]
        for i, p in enumerate(graph.patches):
            for x, y in p.coord_list():
                coord_map[x][y] = i
        x1, y1, x2, y2 = fit_bounding_box(graph.raw_frame, (tlc, (tlc[0] + sx, tlc[1] + sy)))
        return [graph.patches[i] for i in np.unique(coord_map[x1:x2, y1:y2])]


