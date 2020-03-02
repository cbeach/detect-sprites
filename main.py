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
from sprites.find import load_sprites

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

if __name__ == '__main__':
    sky = [248, 148, 88]
    sprite_dir = './sprites/sprites/SuperMarioBros-Nes'
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    ss = SpriteSet(game=game, sprite_dir=sprite_dir, ds=ds)
    sys.exit(0)
    hashes = ss.get_all_hashes()
    fg1 = FrameGraph(cv2.imread('sprites/test/test_cases/SuperMarioBros-Nes/sprites/little_mario_run-1.png', cv2.IMREAD_UNCHANGED), game='SuperMarioBros-Nes', bg_color=sky, ds=ds)
    fg2 = FrameGraph(cv2.imread('sprites/test/test_cases/SuperMarioBros-Nes/sprites/little_mario_run-2.png', cv2.IMREAD_UNCHANGED),  game='SuperMarioBros-Nes', bg_color=sky, ds=ds)
    fg2b = FrameGraph(cv2.imread('sprites/test/test_cases/SuperMarioBros-Nes/sprites/little_mario_run-2.png', cv2.IMREAD_UNCHANGED),  game='SuperMarioBros-Nes', ds=ds)
    fg3 = FrameGraph(cv2.imread('sprites/test/test_cases/SuperMarioBros-Nes/sprites/little_mario_run-3.png', cv2.IMREAD_UNCHANGED), game='SuperMarioBros-Nes', ds=ds)

    #img = cv2.imread('sprites/test/test_cases/SuperMarioBros-Nes/sprites/little_mario_run-3.png', cv2.IMREAD_UNCHANGED)
    #sx, sy, _ = img.shape
    #sky_arr = np.array(sky, dtype=np.uint8)
    #for x, row in enumerate(img[:, :, :3]):
    #    for y, pixel in enumerate(row):
    #        if np.array_equiv(pixel, sky_arr):
    #            img[x][y][3] = 0
    #        else:
    #            img[x][y][3] = 255

    #cv2.imwrite('./little_mario_run-3.png', img)
    #print(img)
    #print(fg3.raw_frame.shape)
    #print(fg3.alpha_chan)
    #print(fg3.alpha_chan * 255)
    #show_image(fg3.alpha_chan * 255, scale=16)

    sg1 = fg1.subgraphs()
    sg2 = fg2.subgraphs()
    sg3 = fg3.subgraphs()

    cv2.imwrite('temp1.png', sg1[0].to_image(border=0))
    cv2.imwrite('temp2.png', sg2[0].to_image(border=0))
    cv2.imwrite('temp3.png', sg3[0].to_image(border=0))
    sys.exit(0)
    i1 = cv2.imread('./17-2.png')
    i2 = cv2.imread('./1492-2.png')

    #print(len(ss.random_unique_patches()), ss.random_unique_patches())
    #ss.is_match(i1, ((0, 0), (16, 16)), 0)
