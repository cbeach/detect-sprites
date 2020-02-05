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

from sprites.sprite import Sprite
from sprites.patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes
from sprites.patch_graph import FrameGraph, Graphlet
from sprites.sprite_util import show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors, ensure_dir, save_partially_processed_playthrough, get_palette, migrate_play_through
import sprites.db
from sprites.db.data_store import DataStore
from sprites.db.models import NodeM, PatchM

ds = DataStore('./sprites/db/sqlite.db', games_path='./sprites/games.json', echo=False)

ig = cv2.imread('./sprites/test/images/ground.png')
ir = cv2.imread('./sprites/test/images/repeating_ground.png')

fgg = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
fgg1 = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
fgr = FrameGraph(ir, ds=ds)

dfgg = FrameGraph(ig, indirect=False, bg_color=[248, 148, 88], ds=ds)
dfgg1 = FrameGraph(ig, indirect=False, ds=ds)
dfgr = FrameGraph(ir, indirect=False, ds=ds)
fgt = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 124, indirect=False, ds=ds)

def initialize_ground_sprite():
    s_dir = f'./sprites/sprites'
    g_dir = f'./sprites/sprites/SuperMarioBros-Nes'
    isprite = Sprite(graphlet=fgg.subgraphs()[0], bg_color=tuple(fgg.bg_color))
    dsprite = Sprite(graphlet=dfgg.subgraphs()[0], bg_color=tuple(dfgg.bg_color))
    isprite.save(f'{s_dir}/ground0')
    dsprite.save(f'{s_dir}/ground1')
    isprite.save(f'{g_dir}/0')
    dsprite.save(f'{g_dir}/1')

def load_sprites(sprite_dir):
    sprites = []
    for path in glob(f'{sprite_dir}/*'):
        extension = path.split('.')[-1]
        sprites.append(Sprite(path=path))

    return sprites

def process_graph(graph, sprites):
    imgs = []
    for sprite in sprites:
        #imgs.append(sprite.remove_from_frame(graph, graph.raw_frame.copy()))
        show_image(sprite.remove_from_frame(graph, graph.raw_frame.copy()), scale=3.0)
    return

    bgc = np.array(graph.bg_color, dtype='uint8')
    new_img = graph.raw_frame.copy()
    for img in imgs:
        for x, row in enumerate(img):
            for y, pix in enumerate(row):
                if np.array_equal(pix, bgc):
                    new_img[x][y] = bgc

    return {
        #'sprites': confirm_sprites(graph, sprites),
        'image': new_img,
    }

def confirm_sprites(graph, sprites):
    not_sprite = []
    for graphlet in graph.subgraphs():
        sprite = Sprite(graphlet=graphlet, bg_color=tuple(graph.bg_color))
        if sprite not in sprites and sprite not in not_sprite:
            resp = graphlet.ask_if_sprite()
            if resp == ord('y'):
                sprites.append(sprite)
            elif resp == 27:
                return sprites
            else:
                not_sprite.append(sprite)
    return sprites


def mine(play_number, game='SuperMarioBros-Nes'):
    play_through_data = migrate_play_through(get_playthrough(play_number, game), play_number, game)
    play_through = play_through_data['partial'] if 'partial' in play_through_data else play_through_data['raw']

    game_dir = f'./sprites/sprites/{game}'
    ensure_dir(game_dir)
    sprites = load_sprites(game_dir)

    start = time.time()
    ifg = FrameGraph(play_through[0], game='SuperMarioBros-Nes', play_num=play_number, frame_num=0, indirect=True, ds=ds)
    processed = process_graph(ifg, sprites[:1])
    sprites = processed['sprites']
    show_image(processed['image'])

    return
    for i, frame in enumerate(play_through[:1]):
        start = time.time()
        ifg = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=True, ds=ds)
        dfg = FrameGraph(frame, indirect=False, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, ds=ds)

        sprites = process_graph(ifg, sprites)
        sprites = process_graph(dfg, sprites)
        for j, sprite in enumerate(sprites):
            sprite.save(f'{game_dir}/{j}')


def main():
    Patch.init_patch_db(ds=ds)
    initialize_ground_sprite()
    mine(1000, 'SuperMarioBros-Nes')

if __name__ == '__main__':
    main()
