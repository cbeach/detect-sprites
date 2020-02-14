#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict
import itertools
import sys
import time

import cv2
from numba import jit
import numpy as np

from .patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes
from .patch_graph import FrameGraph
from .sprite_util import show_images, show_image
from .db.data_store import DataStore

ds = DataStore('sprites/db/sqlite.db', games_path='./sprites/games.json', echo=False)
r, g, b, w, bl = np.array([0, 0, 255]), np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([255, 255, 255]), np.array([0, 0, 0]),

def pairs_equal(a, b):
    return (hash(a[0]) == hash(b[0]) and hash(a[1]) == hash(b[1]) and
            a[0].get_relative_offset(a[1]) == b[0].get_relative_offset(b[1]))

def matching_patches(node, graph):
    return [n for n in graph.patches if hash(n) == hash(node)]

def find_feasible_root_pairs(sprite, graph):
    potential_root_pairs = []
    for root in sprite.patches:
        potential_root_pairs.extend([(root, j) for j in root.get_neighbors()])
    root_pair_set = {(hash(r1), hash(r2)): (r1, r2) for r1, r2 in potential_root_pairs}
    root_pairs = sorted(list(root_pair_set.values()), key=lambda p: p[0].offset_hash(), reverse=True)
    feasible = []
    candidates = [matching_patches(i, graph) for i in sprite.patches]
    for i, p in enumerate(candidates):
        cand_pairs = []
        for j in p:
            for k in j.get_neighbors():
                cand_pairs.append((j, k))

        root_pairs.reverse()
        for j, rp in enumerate(root_pairs):
            for k, cp in enumerate(cand_pairs):
                if pairs_equal(rp, cp):
                    feasible.append(rp)
                else:
                    if graph.indirect != sprite.indirect:
                        #print('graph.indirect != sprite.indirect')
                        #sys.exit(0)
                        pass
                    #print(hash(rp[0]) == hash(cp[0]) and hash(rp[1]) == hash(cp[1]) and
                    #            rp[0].get_relative_offset(rp[1]) == cp[0].get_relative_offset(cp[1]))
                    if hash(rp[0]) == hash(cp[0]) and hash(rp[1]) == hash(cp[1]):
                        pass
                        #print('hashes equal')
                        #print(rp[0].get_relative_offset(rp[1]))
                        #print(cp[0].get_relative_offset(cp[1]))
                        #print('hashes not equal')
                        #show_images(
                        #    (
                        #        rp[0].fill_patch(rp[1].fill_patch(sprite.raw_frame, color=[255, 0, 0])),
                        #        cp[0].fill_patch(cp[1].fill_patch(graph.raw_frame, color=[255, 0, 0])),
                        #    ), scale=3)

    return list(set(feasible))

def find_pairs_in_graph(pair, graph):
    ach, bch = hash(pair[0]), hash(pair[1])
    found = []
    for p in graph.patches:
        for n in p.get_neighbors():
            if pairs_equal(pair, (p, n)) is True:
                found.append((p, n))
    return found

def best_pairs(pairs, graph):
    p2 = [find_pairs_in_graph(p, graph) for p in pairs]
    flat = list(itertools.chain(*p2))
    filtered_by_size = filter(lambda p: p[0].area() + p[1].area() > 5, flat)
    l = [len(p) for p in filtered_by_size]
    if len(l) == 0:
        return None, None
    ind = l.index(max(l))
    return pairs[ind], p2[ind]

def fit_bounding_box(img, bb):
    l, r = bb
    lx, ly = l
    rx, ry = r
    bblx = 0 if lx < 0 else lx
    bbrx = img.shape[0] if rx >= img.shape[0] else rx
    bbly = 0 if ly < 0 else ly
    bbry = img.shape[1] if rx >= img.shape[1] else ry
    return bblx, bbly, bbrx, bbry

def cull_sprites(graph, sprite, coords, image):
    #show_image(sprite.raw_frame, scale=8)
    for i, c in enumerate(coords):  # top left, bottom right corners
        l, r = c
        lx, ly = l
        rx, ry = r
        sx, sy = rx - lx, ry - ly

        bblx, bbly, bbrx, bbry = fit_bounding_box(image, (l, r))
        nlx, nly = bblx - lx, bbly - ly
        nrx, nry = sprite.raw_frame.shape[0] - (rx - bbrx), sprite.raw_frame.shape[1] - (ry - bbry)
        s_img = sprite.raw_frame.copy()[nlx:nrx, nly:nry, :]
        for x, row in enumerate(image[bblx:bbrx]):
            for y, pix in enumerate(row[bbly:bbry]):
                if sprite.alpha_chan[x][y] > 0 and np.array_equal(pix, s_img[x][y]):
                    #print(x, y)
                    pix[0] = graph.bg_color[0]
                    pix[1] = graph.bg_color[1]
                    pix[2] = graph.bg_color[2]

    return image

def get_sprite_coords(graph, sprite, ref, anchors):
    coords = []
    ox, oy = ref[0].bounding_box[0]
    sx, sy = sprite.shape[:2]
    for a, b in anchors:
        aox, aoy = a.bounding_box[0]
        coords.append(((aox - ox,  aoy - oy), (aox - ox + sx, aoy - oy + sy)))
    return coords

def find_and_cull(graph, sprites):
    image = graph.raw_frame.copy()
    graphlets = graph.subgraphs()
    for i, sprite in enumerate(sprites):
        if len(sprite.patches) == 1:
            for graphlet in graphlets:
                if len(graphlet.nodes) == 1 and hash(graphlet.nodes[0]) == hash(sprite.patches[0]):
                    image = cull_sprites(graph, sprite, [graphlet.bb], image)
        else:
            pairs = find_feasible_root_pairs(sprite, graph)
            if len(pairs) == 0:
                continue
            ref_pair, anchors = best_pairs(pairs, graph)
            if ref_pair is None or anchors is None:
                continue
            coords = get_sprite_coords(graph, sprite.raw_frame, ref_pair, anchors)
            image = cull_sprites(graph, sprite, coords, image)

    return image
