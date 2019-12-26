from glob import glob
import gzip
import itertools
import pickle
import random
import sys
import time

import cv2
import jsonpickle
import numpy as np

from patch_graph import FrameGraph
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors
import db
from db.data_store import DataStore
from db.models import FrameGraphM, NodeM, PatchM


if __name__ == '__main__':
    ds = DataStore('temp.db', games_path='./games.json', echo=False)
    fg = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=True, ds=ds)

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
