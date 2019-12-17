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
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors
import db
from db.data_store import DataStore



if __name__ == '__main__':
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
