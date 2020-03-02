#!/usr/bin/env python
# coding: utf-8

from collections import namedtuple
from glob import glob
import itertools
import os
import sys
import time

import cv2
from numba import jit
import numpy as np

from .patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes
from .patch_graph import FrameGraph
from .sprite_util import show_images, show_image, neighboring_points
from .db.data_store import DataStore

ds = DataStore('sprites/db/sqlite.db', games_path='./sprites/games.json', echo=False)
r, g, b, w, bl = np.array([0, 0, 255]), np.array([0, 255, 0]), np.array([255, 0, 0]), np.array([255, 255, 255]), np.array([0, 0, 0]),

def merge_images(img1, img2, bg_color):
    merged = np.zeros_like(img1)
    for x, row in enumerate(img1):
        for y, pixel in enumerate(row):
            if np.array_equal(img1[x][y], bg_color) or np.array_equal(img2[x][y], bg_color):
                merged[x][y] = bg_color
            else:
                merged[x][y] = img1[x][y]

    return merged

@jit(nopython=True)
def normalize_image(img, indirect=True):
    sx, sy, sc = img.shape
    normed = np.zeros(img.shape[:-1], dtype=np.uint8)
    mask = np.zeros(img.shape[:-1], dtype=np.bool8)
    index = 1
    for x in range(len(mask)):
        row = mask[0]
        for y, visited in enumerate(row):
            if visited is True:
                continue
            stack = [(x, y)]
            while len(stack) > 0:
                cx, cy = stack.pop()  # current_x, current_y
                if not mask[cx][cy]:
                    normed[cx][cy] = index

                nbr_coords = neighboring_points(cx, cy, img, indirect)
                for nx, ny in nbr_coords:
                    if mask[nx][ny] == False and np.array_equal(img[cx][cy], img[nx][ny]):
                        stack.append((nx, ny))
                mask[cx][cy] = True

            index += 1
    return normed

def unique_sprites(graphs):
    indirected = [FrameGraph(i.raw_frame, indirect=True, ds=ds) for i in graphs if i.indirect is False]
    normed = [normalize_image(i.raw_frame).flatten() for i in graphs]
    shape_normed = [np.array((*graphs[i].raw_frame.shape, *fn)) for i, fn in enumerate(normed)]
    unique_sprites = []
    unique_indices = []
    for i, sn in enumerate(shape_normed):
        if not any([np.array_equal(sn, j) for j in unique_sprites]):
            unique_sprites.append(sn)
            unique_indices.append(i)

    return graphs

def load_sprites(sprite_dir='./sprites/sprites/SuperMarioBros-Nes', game='SuperMarioBros-Nes', ds=None):
    isprites = []
    dsprites = []
    for i, path in enumerate(glob(f'{sprite_dir}/*')):
        extension = path.split('.')[-1]
        if extension == 'png':
            isprites.append(FrameGraph.from_path(path, game=game, indirect=True, ds=ds))
            dsprites.append(FrameGraph.from_path(path, game=game, indirect=False, ds=ds))

    return {
        True: isprites,
        False: dsprites,
    }

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

def confirm_sprites(src_dir, dest_dir):
    PossibleSprite = namedtuple('Sprite', ['path', 'img'])
    sprite_count = len(list(glob(f'{dest_dir}/*.png')))
    possible_sprites = [PossibleSprite(path=path, img=cv2.imread(path, cv2.IMREAD_UNCHANGED))
                        for path in glob(f'{src_dir}/*.png')]
    normed = [tuple(normalize_image(ps.img).flatten()) for ps in possible_sprites]
    unique_possible_sprites = {n: sprite for n, sprite in zip(normed, possible_sprites)}.values()
    print(len(unique_possible_sprites))
    for i, ps in enumerate(unique_possible_sprites):
        sx, sy, sd = ps.img.shape
        scale = 720 // max((sx, sy))
        for x, row in enumerate(ps.img):
            for y, pixel in enumerate(row):
                if ps.img[x][y][3] == 0:
                    ps.img[x][y][0] = 0
                    ps.img[x][y][1] = 92
                    ps.img[x][y][2] = 0

        print(f'{i} of {len(unique_possible_sprites)}: shape: {ps.img.shape}')
        resp = show_image(ps.img, scale=scale)
        if resp == ord('y') or resp == ord('Y'):
            cv2.imwrite(f'{dest_dir}/{sprite_count + i}.png', ps.img)
            os.remove(ps.path)
        elif resp == ord('n') or resp == ord('N'):
            os.remove(ps.path)
        elif resp == 27:  # ESC key
            break
        else:
            continue
    for ps in possible_sprites:
        try:
            os.remove(ps.path)
        except FileNotFoundError as err:
            print(f'file not found: {ps.path}')
