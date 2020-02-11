import dill
from glob import glob
import gzip
import itertools
import json
import pickle
import random
import sys
import time

import cv2
import jsonpickle
import numpy as np
from termcolor import cprint
from numba import jit

from sprites.sprite import Sprite
from sprites.patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes
from sprites.patch_graph import FrameGraph, Graphlet
from sprites.sprite_util import neighboring_points, show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors, ensure_dir, save_partially_processed_playthrough, get_palette, migrate_play_through, palettize_image
import sprites.db
from sprites.db.data_store import DataStore
from sprites.db.models import NodeM, PatchM
from find_sprites import find_and_cull

ds = DataStore('./sprites/db/sqlite.db', games_path='./sprites/games.json', echo=False)
Patch.init_patch_db(ds=ds)

#ig = cv2.imread('./sprites/test/images/ground.png')
#ir = cv2.imread('./sprites/test/images/repeating_ground.png')
#
#fgg = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
#fgg1 = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
#fgr = FrameGraph(ir, ds=ds)
#fgt = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 124, indirect=True, ds=ds)
#
#dfgg = FrameGraph(ig, indirect=False, bg_color=[248, 148, 88], ds=ds)
#dfgg1 = FrameGraph(ig, indirect=False, ds=ds)
#dfgr = FrameGraph(ir, indirect=False, ds=ds)
#dfgt = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 124, indirect=False, ds=ds)
#it = fgt.raw_frame.copy()

sky = [248, 148, 88]

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

    return [graphs[i].raw_frame for i in unique_indices]

def load_sprites(sprite_dir, game, bg_color):
    isprites = []
    dsprites = []
    for i, path in enumerate(glob(f'{sprite_dir}/*')):
        extension = path.split('.')[-1]
        if extension == 'png':
            isprites.append(FrameGraph.from_path(path, game=game, bg_color=bg_color, indirect=True, ds=ds))
            dsprites.append(FrameGraph.from_path(path, game=game, bg_color=bg_color, indirect=False, ds=ds))

    return {
        True: isprites,
        False: dsprites,
    }

def process_frame(frame, play_number, frame_number, sprites=None, **kwargs):
    cont = True
    while cont:
        old_sprite_counts = {
            True: len(sprites[True]),
            False: len(sprites[False]),
        }

        igraph = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=frame_number, indirect=True, ds=ds)
        dgraph = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=frame_number, indirect=False, ds=ds)
        old_i_img = igraph.raw_frame
        old_d_img = dgraph.raw_frame

        try:
            sprites[True] = confirm_sprites(igraph, sprites, True)
            sprites[False] = confirm_sprites(dgraph, sprites, False)
        except EOFError:
            return

        i_img = find_and_cull(igraph, sprites[True])
        d_img = find_and_cull(igraph, sprites[False])

        if (len(sprites[True]) == old_sprite_counts[True]
            and len(sprites[False]) == old_sprite_counts[False]
            and np.array_equal(old_i_img, i_img)
            and np.array_equal(old_d_img, d_img)):
            cont = False


    sky = np.array([248, 148, 88], dtype='uint8')
    new_img = frame.copy()
    for i in (i_img, d_img):
        for x, row in enumerate(i):
            for y, pix in enumerate(row):
                if np.array_equal(pix, sky):
                    new_img[x][y][0] = sky[0]
                    new_img[x][y][1] = sky[1]
                    new_img[x][y][2] = sky[2]
    show_image(new_img, scale=3)

    return {
        'sprites': sprites,
        #'image': new_img,
    }

def confirm_sprites(graph, sprites, indirect, supervised=True, not_sprites=None):
    other_sprites = sprites[not indirect]
    sprites = sprites[indirect]
    not_sprite = [] if not_sprites is None else not_sprites
    new_sprites = []
    for graphlet in graph.subgraphs():
        sprite = graphlet.clipped_frame
        if sprite.shape[0] * sprite.shape[1] > 32 ** 2:
            continue

        query = not any(
              [np.array_equal(i.subgraphs()[0].clipped_frame, sprite) for i in sprites]
            + [np.array_equal(i.subgraphs()[0].clipped_frame, sprite) for i in other_sprites]
            + [np.array_equal(i, sprite) for i in new_sprites]
            + [np.array_equal(i, sprite) for i in not_sprite])
        if query is True:
            resp = graphlet.ask_if_sprite(parent_img=graph.raw_frame)
            if resp == ord('y'):
                new_sprites.append(sprite)
            elif resp == ord('n'):
                not_sprite.append(sprite)
            elif resp == ord('q'):
                break
            elif resp == 27:
                raise EOFError('user finished')

    return sprites \
        + [FrameGraph(s, bg_color=graph.bg_color, indirect=indirect, ds=ds) for s in new_sprites], not_sprites

def mine(play_number, game='SuperMarioBros-Nes'):
    play_through_data = migrate_play_through(get_playthrough(play_number, game), play_number, game)
    play_through = play_through_data['partial'] if 'partial' in play_through_data else play_through_data['raw']

    game_dir = f'./sprites/sprites/{game}'
    ensure_dir(game_dir)
    sprites = load_sprites(game_dir, game, sky)

    #for i, frame in enumerate(random.choices(play_through, k=2)):
    for i, frame in enumerate(play_through[:1]):
        ret_val = process_frame(frame, play_number=play_number, frame_number=i, **params)

    for j, sprite in enumerate([j for j in ret_val['sprites'][True]]):
        show_image(sprite.raw_frame, scale=8.0)
        cv2.imwrite(f'{game_dir}/{j}.png', sprite.raw_frame)

def find(play_number, game='SuperMarioBros-Nes', sample=None, random_pop=False, supervised=True):
    play_through_data = migrate_play_through(get_playthrough(play_number, game), play_number, game)
    play_through = play_through_data['partial'] if 'partial' in play_through_data else play_through_data['raw']

    game_dir = f'./sprites/sprites/{game}'
    ensure_dir(game_dir)
    sprites = load_sprites(game_dir, game, sky)
    not_sprites = []
    if random_pop is True:
        play_through = random.shuffle(play_through)

    for i, frame in enumerate(play_through[:sample]):
        igraph = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=True, ds=ds)
        dgraph = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=False, ds=ds)

        try:
            sprites[True], not_sprites = confirm_sprites(igraph, sprites, True, not_sprites=not_sprites)
            sprites[False], not_sprites = confirm_sprites(dgraph, sprites, False, not_sprites=not_sprites)
        except EOFError:
            return
        print('frame number:', i)

    us = unique_sprites(sprites[True] + sprites[False])
    print('number of unique sprites:', len(us))
    for i, sprite in enumerate([j for j in us]):
        show_image(sprite, scale=8.0)
        cv2.imwrite(f'{game_dir}/{i}.png', sprite)

def cull(play_number, game='SuperMarioBros-Nes'):
    play_through_data = migrate_play_through(get_playthrough(play_number, game), play_number, game)
    play_through = play_through_data['partial'] if 'partial' in play_through_data else play_through_data['raw']

    game_dir = f'./sprites/sprites/{game}'
    ensure_dir(game_dir)
    s = load_sprites(game_dir, game, sky)
    sprites = {
        True: s[True],
        False: s[False],
    }
    params = {
        'sprites': sprites,
    }

def show_sprites():
    game='SuperMarioBros-Nes'
    game_dir = f'./sprites/sprites/{game}'
    sprites = load_sprites(game_dir, game, sky)[True]
    show_images([s.raw_frame for s in sprites], scale=8)

def main():
    find(1000, 'SuperMarioBros-Nes', sample=50)

if __name__ == '__main__':
    main()

