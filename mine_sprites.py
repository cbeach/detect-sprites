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
from sprites.sprite_util import show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors, ensure_dir
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

def load_sprites(sprite_dir):
    sprites = []
    for path in glob(f'{sprite_dir}/*'):
        extension = path.split('.')[-1]
        sprites.append(Sprite(path=path))

    return sprites

def is_sprite(graphlet):
    resp = graphlet.is_sprite()

def process_graph(graph, sprites):
    return confirm_sprites(graph, sprites)

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
    play_through = get_playthrough(play_number, game)
    game_dir = f'./sprites/sprites/{game}'
    ensure_dir(game_dir)
    sprites = load_sprites(game_dir)

    for i, frame in enumerate(play_through[:1]):
        start = time.time()
        if np.array_equal(frame[0][0], np.array([88, 148, 248], dtype='uint8')):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ifg = FrameGraph(frame, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=True, ds=ds)
        dfg = FrameGraph(frame, indirect=False, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, ds=ds)

        sprites = process_graph(ifg, sprites)
        sprites = process_graph(dfg, sprites)
        for j, sprite in enumerate(sprites):
            sprite.save(f'{game_dir}/{j}')


def main():
    mine(1000, 'SuperMarioBros-Nes')


if __name__ == '__main__':
    main()
