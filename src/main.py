from glob import glob
import gzip
import itertools
import pickle
import random
import sys
import time

import jsonpickle
import cv2

from patch_graph import FrameGraph
from data_store import PatchDB
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors



if __name__ == '__main__':
    #pt = list(load_indexed_playthrough(1000, count=1))
    #direct = [i['direct'] for i in pt]
    #indirect = [i['indirect'] for i in pt]
    #direct_sgs = []
    #hsh_set = {}

    dpg = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=False)
    sgs = dpg.subgraphs()
    esc_key = 27
    for i, sg in enumerate(sgs):
        resp, sprite = sg.ask_if_sprite(bg_color=(255, 0, 0))
        #resp = sg.show()
        if resp == 27:
            break

        #direct_sgs.append(temp)
    #ipg = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=True)
    #dpg = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=False)
    #dsg, dohs = dpg.subgraphs()
    #print(f'len(dpg.offset_hashes): {len(dpg.offset_hashes)}')
    #print(f'len(list(itertools.chain(*dsg))): {len(list(itertools.chain(*dsg)))}')
    #print(f'len(list(itertools.chain(*dohs))): {len(list(itertools.chain(*dohs)))}')
    #print(f'len(set(itertools.chain(*dohs))): {len(set(itertools.chain(*dohs)))}')
    #dohs = sorted(list(set(itertools.chain(*dohs))))
    #dpgohs = sorted(dpg.offset_hashes)
    #count = 0
    #for i in zip(dpgohs, dohs):
    #    if i[0] != i[1]:
    #        count += 1

    #print(f'count 1: {count}')

    #for i in dsg:
    #    print(f'area: {dpg.subgraph_area(i)}')
    #    show_image(dpg.fill_subgraph(i), scale=2)
