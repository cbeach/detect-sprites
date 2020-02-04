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

from sprites.patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes
from sprites.patch_graph import FrameGraph
from sprites.sprite_util import show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors
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


def parse_and_store_play_through(play_number):
    ds = DataStore('db/sqlite.db', games_path='./games.json', echo=False)
    #ds.initialize()
    Patch.init_patch_db(ds)
    sess = ds.Session()

    play_through = get_playthrough(play_number)
    start_time = time.time()

    for i, img in enumerate(play_through[:10]):
        frame_start = time.time()
        print(f'frame {i}')
        parse_start = time.time()
        print(f'  indirect')
        ifg = FrameGraph(frame=img, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=True, ds=ds)
        print(f'  direct')
        dfg = FrameGraph(frame=img, game='SuperMarioBros-Nes', play_num=play_number, frame_num=i, indirect=False, ds=ds)
        print(f'  frame parsing: {time.time() - parse_start}')

        linking_start = time.time()
        print(f'  linking nodes')
        try:
            ifg.add_neighbors_to_nodes()
            dfg.add_neighbors_to_nodes()
        except AttributeError:
            cprint(f'error processing playthrough number {play_number}, frame_number {i}')
        print(f'  linking: {time.time() - linking_start}')


        store_start = time.time()
        print(f'  storing graphs')
        try:
            ifg.store(ds)
            dfg.store(ds)
        except AttributeError:
            show_image(img)
            cprint(f'error processing playthrough number {play_number}, frame_number {i}')
        print(f'  store: {time.time() - store_start}')
        print(f'elapsed time for frame {i}: {time.time() - frame_start}')
        print(f'total elapsed: {time.time() - start_time}')
        print('---------------------------------------------------\n')

    print(f'number of patches: {len(Patch._PATCHES)}')
    for i, p in enumerate(Patch._PATCHES.values()):
        print(f'storing patch {i}')
        p.store(sess)
    sess.commit()
    cprint(f'finished processing playthrough {play_number}: \n  elapsed time: {time.time() - start_time}', 'green')


def test_isomorphism():
    ds = DataStore('./sprites/db/sqlite.db', games_path='./sprites/games.json', echo=False)
    grnd_img = cv2.imread('./sprites/ground.png')
    ifg_grnd = FrameGraph(grnd_img, indirect=False, bg_color=[248, 148, 88], ds=ds)
    rep_grnd_img = cv2.imread('./sprites/repeating_ground.png')
    rep_ifg_grnd = FrameGraph(rep_grnd_img, indirect=False, ds=ds)
    degen_rep_grnd_img = cv2.imread('./sprites/degenerate_repeating_ground.png')
    degen_rep_ifg_grnd = FrameGraph(degen_rep_grnd_img, indirect=False, ds=ds)
    cont = True
    test_relative_offset()
    sys.exit(0)
    for ic, i in enumerate(rep_ifg_grnd.patches):
        for jc, j in enumerate(ifg_grnd.patches):
            print(jc, i.ch())
            show_image(j.fill_patch(grnd_img), scale=8.0)
            continue
            if i == j:
                try:
                    print(ic, i.neighborhood_similarity(j))
                    img = i.fill_neighborhood(rep_ifg_grnd.raw_frame)
                    sprt = j.fill_neighborhood(grnd_img)
                    resp = show_images((img, sprt), scale=8.0)
                    if resp == 27:
                        sys.exit(0)
                    #cont = False
                    break
                except AssertionError as e:
                    print('exception')
                    print(i, bin(i.master_hash(color=True)), i.color, [k.master_hash(color=True) for k in i.neighbors])
                    print(j, bin(j.master_hash(color=True)), j.color, [k.master_hash(color=True) for k in j.neighbors])
                    a = i.fill_patch(rep_grnd_img)
                    for k in i.neighbors:
                        a = k.fill_patch(a, color=(255, 0, 0))
                    b = j.fill_patch(grnd_img)
                    for k in j.neighbors:
                        b = k.fill_patch(b, color=(255, 0, 0))
                    show_images((a, b), scale=12.0)
                    raise(e)
        if cont is False:
            break


if __name__ == '__main__':
    fgt.remove_nodes(fgt.patches[10:20])
    sys.exit(0)

    fg.store()
    sess = ds.Session()
    sess.query(NodeM).all()
    for i in sess.query(NodeM).all():
        print('main:', i)
    print('NodeM.count():', sess.query(NodeM).count())

    for i in sess.query(FrameGraphM).all():
        print('main:', i)
    print('FrameGraphM.count():', sess.query(FrameGraphM).count())
    print('Done')
    print(fg.adjacency_matrix.shape)
    size = fg.adjacency_matrix.shape[0] * fg.adjacency_matrix.shape[1]
    o_size = fg.offset_adjacency_matrix.shape[0] * fg.offset_adjacency_matrix.shape[1]
    print('size', size, size / 8.0)
    print('o_size', o_size, o_size / 8.0)
    print('non_zero', np.count_nonzero(fg.adjacency_matrix))
    print('o_non_zero', np.count_nonzero(fg.offset_adjacency_matrix))

    sys.exit(0)

    ds = DataStore()
    ds.initialize_db('./games.json')

    sys.exit(0)
    dpg = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=False)
    sgs = dpg.subgraphs()
    esc_key = 27
    for i, sg in enumerate(sgs):
        resp, sprite = sg.ask_if_sprite(bg_color=(255, 0, 0))
        #resp = sg.show()
        if resp == 27:
            break
