from glob import glob
import gzip
import itertools
import pickle
import random
import sys
import time

import jsonpickle
import cv2

from patch_graph import PatchGraph
from data_store import PatchDB
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough



if __name__ == '__main__':
    pt = list(load_indexed_playthrough(1000, count=100))
    direct = [i['direct'] for i in pt]
    indirect = [i['indirect'] for i in pt]
    direct_sgs = []
    hsh_set = {}
    for i in direct:
        sg = i.isolate_offset_subgraphs()
        hashes = []
        for j in sg:
            for k in j:
                t1 = sorted(map(lambda a: hash(a), k))
                t2 = tuple(t1)
                hsh_set[t2] = k
        #direct_sgs.append(temp)
    #ipg = PatchGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=True)
    #dpg = PatchGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=False)
    #dsg, dohs = dpg.isolate_offset_subgraphs()
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
