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
sky = [248, 148, 88]

def load_sprites(sprite_dir, game, bg_color):
    sprites = []
    for path in glob(f'{sprite_dir}/*'):
        extension = path.split('.')[-1]
        sprites.append(FrameGraph.from_path(path, game=game, bg_color=bg_color, indirect=True, ds=ds))
        sprites.append(FrameGraph.from_path(path, game=game, bg_color=bg_color, indirect=False, ds=ds))

    return sprites

def process_graph(graph, sprites):
    #imgs = []
    #for sprite in sprites:
    #    #imgs.append(sprite.remove_from_frame(graph, graph.raw_frame.copy()))
    #    show_image(sprite.remove_from_frame(graph, graph.raw_frame.copy()), scale=3.0)
    #return

    #bgc = np.array(graph.bg_color, dtype='uint8')
    #new_img = graph.raw_frame.copy()
    #for img in imgs:
    #    for x, row in enumerate(img):
    #        for y, pix in enumerate(row):
    #            if np.array_equal(pix, bgc):
    #                new_img[x][y] = bgc

    return {
        'sprites': confirm_sprites(graph, sprites),
        #'image': new_img,
    }

def confirm_sprites(graph, sprites):
    not_sprite = []
    new_sprites = []
    for graphlet in graph.subgraphs():
        sprite = graphlet.clipped_frame
        query = not any([np.array_equal(i.subgraphs()[0].clipped_frame, sprite) for i in sprites] + [np.array_equal(i, sprite) for i in new_sprites] + [np.array_equal(i, sprite) for i in not_sprite])
        if query is True:
            resp = graphlet.ask_if_sprite(parent_img=graph.raw_frame)
            if resp == ord('y'):
                new_sprites.append(sprite)
            elif resp == 27:
                break
            else:
                not_sprite.append(sprite)

    return sprites \
        + [FrameGraph(s, bg_color=graph.bg_color, indirect=True, ds=ds) for s in new_sprites] \
        + [FrameGraph(s, bg_color=graph.bg_color, indirect=False, ds=ds) for s in new_sprites]



def mine(play_number, game='SuperMarioBros-Nes'):
    play_through_data = migrate_play_through(get_playthrough(play_number, game), play_number, game)
    play_through = play_through_data['partial'] if 'partial' in play_through_data else play_through_data['raw']

    game_dir = f'./sprites/sprites/{game}'
    ensure_dir(game_dir)
    ret_val = {
        'sprites': load_sprites(game_dir, game, sky),
    }

    print('random.choices(play_through, k=2)', len(random.choices(play_through, k=2)))
    #for i, frame in enumerate(random.choices(play_through, k=2)):
    for i, frame in enumerate(play_through[:1]):
        start = time.time()
        ifg = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=True, ds=ds)
        dfg = FrameGraph(frame, indirect=False, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, ds=ds)

        ret_val = process_graph(ifg, ret_val['sprites'])
        ret_val = process_graph(dfg, ret_val['sprites'])

    for j, sprite in enumerate([j for j in ret_val['sprites'] if j.indirect is True]):
        cv2.imwrite(f'{game_dir}/{j}.png', sprite.raw_frame)

def main():
    Patch.init_patch_db(ds=ds)
    mine(1000, 'SuperMarioBros-Nes')

if __name__ == '__main__':
    main()
