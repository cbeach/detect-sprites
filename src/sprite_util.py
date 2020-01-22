from collections import defaultdict
from glob import glob
import gzip
import itertools
import json
import os
import pickle
import random
import time

import numpy as np
import cv2

DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/home/mcsmash/dev/data'
PLAY_DIR = f'{DATA_DIR}/game_playing/play_data/'
PNG_TMPL = '{DATA_DIR}/game_playing/play_data/{game}/frames/{play_number}-{frame_number}.png'
PICKLE_DIR = './db/SuperMarioBros-Nes/pickle'

def neighboring_points(x, y, arr, indirect=True):
    max_x, max_y = arr.shape[:2]
    neighbors = []
    if x > 0 and y > 0 and indirect is True:
        neighbors.append((x-1, y-1))
    if y > 0:
        neighbors.append((x, y-1))
    if x < max_x - 1 and y > 0 and indirect is True:
        neighbors.append((x+1, y-1))

    if x > 0:
        neighbors.append((x-1, y))
    #if x > 0 and np.array_equal(frame[x][y], frame[x][y]):
    #    neighbors.append((x, y))
    if x < max_x - 1:
        neighbors.append((x+1, y))

    if x > 0 and y < max_y - 1 and indirect is True:
        neighbors.append((x-1, y+1))
    if y < max_y - 1:
        neighbors.append((x, y+1))
    if x < max_x - 1 and y < max_y - 1 and indirect is True:
        neighbors.append((x+1, y+1))

    return neighbors

def show_image(img, scale=1.0):
    cv2.imshow('frame', cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    return cv2.waitKeyEx(0)

def get_image_list(game='SuperMarioBros-Nes', play_number=None):
    print('Loading image file list')
    if play_number is None:
        with open('./image_files', 'r') as fp:
            frame_paths = [i.strip() for i in fp.readlines()]
        random.shuffle(frame_paths)
        return frame_paths
    else:
        return glob(f'{PLAY_DIR}/{game}/{play_number}/*')

def get_frame(game, play_number, frame_number):
    return cv2.imread(PNG_TMPL.format(DATA_DIR=DATA_DIR, game=game, play_number=play_number, frame_number=frame_number))
def get_playthrough(game='SuperMarioBros-Nes', play_number=None):
    if play_number is not None:
        d = np.load(f'{PLAY_DIR}/{game}/{play_number}/frames.npz')
        return d['arr_0']
    else:
        raise TypeError

def load_indexed_playthrough(play_number, count=None):
    db_path = get_db_path(play_number)
    if not ensure_dir(db_path):
        frame_paths = []
    else:
        frame_paths = glob(f'{db_path}/*')

    start = time.time()
    if count is None:
        number_of_frames = len(frame_paths)
    else:
        number_of_frames = count

    for i in range(number_of_frames):
        print(f'loading: {i}')
        with gzip.GzipFile(f'{PICKLE_DIR}/{play_number}/{i}.pickle', 'rb') as fp:
            yield pickle.load(fp)

def get_db_path(play_number, backend='pickle'):
    if backend == 'pickle':
        return f'{PICKLE_DIR}/{play_number}'

def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        return False
    return True

def sort_colors(color_list):
    """
        Don't judge me. My brain isn't working.
    """
    cl = sorted([f'{chr(b)}{chr(g)}{chr(r)}' for b, g, r in color_list])
    return [(ord(b), ord(g), ord(r)) for b, g, r in cl]

def conjugate_numbers(num, seed=0, num_length=None):
    """
        bin(seed) + bin(num)
        seed ends up in most significant bits
    """
    if num_length is None:
        # TODO: Use log you dip
        num_length = len(bin(num)) - 2
    return (seed << num_length) | int(num)

def patch_encoder(patch):
    return {
        'mask': patch.get_mask().astype('uint8').tolist(),
    }

def node_encoder(node, palette=None):
    if palette is None:
        color = node.color.tolist()
    else:
        color = palette.index(tuple(node.color))

    return {
        'tlc': node.bounding_box[0],
        'color': color,
        'patch': patch_encoder(node.patch),
    }

def graph_encoder(frame):
    palette = frame.palette
    am = frame.offset_adjacency_matrix
    nodes = [node_encoder(node, palette=frame.palette) for node in frame.patches if not node.is_frame_edge() and not node.is_background()]
    return {
        'palette': palette,
        'nodes': nodes,
        'adjacency_matrix': frame.offset_adjacency_matrix.astype('uint8').tolist(),
        'shape': frame.raw_frame.shape,
    }

