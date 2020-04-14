from collections import namedtuple
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

from sprites.patch import Patch
from sprites.sprite_set import SpriteSet
from sprites.patch_graph import FrameGraph
from sprites.sprite_util import show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors
from sprites.db.data_store import DataStore
from sprites.find import load_sprites, cull_sprites, fit_bounding_box

ds = DataStore('./sprites/db/sqlite.db', games_path='./sprites/games.json', echo=False)
Patch.init_patch_db(ds)

fgt = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 124, indirect=False, ds=ds)
ig = cv2.imread('./sprites/test/test_cases/SuperMarioBros-Nes/images/ground.png')
ir = cv2.imread('./sprites/test/test_cases/SuperMarioBros-Nes/images/repeating_ground.png')
#
#fgg = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
#fgg1 = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
#fgr = FrameGraph(ir, ds=ds)
#
#dfgg = FrameGraph(ig, indirect=False, bg_color=[248, 148, 88], ds=ds)
#dfgg1 = FrameGraph(ig, indirect=False, ds=ds)
#dfgr = FrameGraph(ir, indirect=False, ds=ds)


def get_frame_count():
    ds = DataStore('db/sqlite.db', games_path='./games.json', echo=False)
    Patch.init_patch_db(ds)

    data = []
    accum = 0
    for play_number in range(1, 17696):
        try:
            play_through = get_playthrough(play_number)
            data.append((play_number, len(play_through), accum, play_through.shape))
            print(data[-1])
            accum += len(play_through)
        except FileNotFoundError as e:
            print('FileNotFoundError: ', play_number)
        except Exception as e:
            print(e)

    with open('frame_count', 'w') as fp:
        json.dump(data, fp, indent=2)

def get_matching_nodes(graph, reference_node):
    return [node for node in graph.patches if hash(node) == hash(reference_node)]

SpriteLocation = namedtuple('SpriteLocation', ['bounding_box', 'path'])

if __name__ == '__main__':
    sky = [248, 148, 88]
    sprite_dir = './sprites/sprites/SuperMarioBros-Nes'
    test_sprite_dir = './sprites/test/test_cases/SuperMarioBros-Nes/sprites/test_sets/sprite_set' 
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    #ss = SpriteSet(game=game, sprite_dir=test_sprite_dir, ds=ds)
    ss = SpriteSet(game=game, sprite_dir=sprite_dir, ds=ds)
    print('sprite count:')
    print('\tTrue:', len(ss.sprites[True]))
    print('\tFalse:', len(ss.sprites[False]))
    pt = get_playthrough(1000, 'SuperMarioBros-Nes')['raw']
    matches = []
    for i, frame in enumerate(pt):
        print(i)
        graphs = {
            True: FrameGraph(frame, bg_color=sky, indirect=True, ds=ds),
            False: FrameGraph(frame, bg_color=sky, indirect=False, ds=ds),
        }
        img = frame.copy()
        for ntype in [True, False]:
            graph = graphs[ntype]
            for j, single in enumerate(ss._single_node_sprites[ntype]):
                sprite = ss.sprites[ntype][single]
                matching_nodes = get_matching_nodes(graph, sprite.patches[0])
                for node in matching_nodes:
                    matches.append(SpriteLocation(bounding_box=node.bounding_box, path=ss.sprites['paths'][j]))
                    img = cull_sprites(graph, sprite, [node.bounding_box], img)

            for j, roots in ss._upi[ntype].items():
                #print(roots)
                sprite = ss.sprites[ntype][j]
                for root_node in roots:
                    matching_nodes = get_matching_nodes(graph, root_node)
                    for node in matching_nodes:
                        #show_image(node.fill_patch(img), scale=3)
                        tlc = (node.bounding_box[0][0] - root_node.bounding_box[0][0], node.bounding_box[0][1] - root_node.bounding_box[0][1])
                        brc = (tlc[0] + sprite.raw_frame.shape[0], tlc[1] + sprite.raw_frame.shape[1])
                        bblx, bbly, bbrx, bbry = fit_bounding_box(graph.raw_frame, (tlc, brc))
                        bb = ((bblx, bbly), (bbrx, bbry))
                        matches.append(SpriteLocation(bounding_box=bb, path=ss.sprites['paths'][j]))
                        img = cull_sprites(graph, sprite, [bb], img)
                        #show_image(img, scale=3)
        cv2.imwrite(f'output/{i}.png', img)
    sys.exit(0)
