from glob import glob
import gzip
import itertools
import pickle
import random
import sys
import time

import jsonpickle
import cv2

from frame import Frame
from patch_graph import PatchGraph
from data_store import PatchDB
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough



if __name__ == '__main__':
    #pl = list(load_indexed_playthrough(1000))
    pg = PatchGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98)
    sg, ohs = pg.isolate_offset_subgraphs()
    print(f'len(pg.offset_hashes): {len(pg.offset_hashes)}')
    print(f'len(list(itertools.chain(*sg))): {len(list(itertools.chain(*sg)))}')
    print(f'len(list(itertools.chain(*ohs))): {len(list(itertools.chain(*ohs)))}')
    print(f'len(set(itertools.chain(*ohs))): {len(set(itertools.chain(*ohs)))}')
    ohs = sorted(list(set(itertools.chain(*ohs))))
    pgohs = sorted(pg.offset_hashes)
    count = 0
    for i in zip(pgohs, ohs):
        if i[0] != i[1]:
            count += 1

    print(f'count 1: {count}')

    for i in sg:
        print(i)
        print()
        show_image(pg.fill_subgraph(i), scale=2)

