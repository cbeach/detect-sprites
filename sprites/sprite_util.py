from collections import defaultdict
from glob import glob
import gzip
import itertools
import json
import math
import os
import pickle
from queue import Queue
from queue import LifoQueue as Stack
import random
import time
from typing import Callable, List, Dict, Any, Tuple

from numba import jit
import numpy as np
import cv2

DATA_DIR = os.environ['DATA_DIR'] if 'DATA_DIR' in os.environ else '/home/mcsmash/dev/data'
PLAY_DIR = f'{DATA_DIR}/game_playing/play_data/'
PNG_TMPL = '{DATA_DIR}/game_playing/play_data/{game}/frames/{play_number}-{frame_number}.png'
PICKLE_DIR = './db/SuperMarioBros-Nes/pickle'

@jit(nopython=True)
def flattened_neighboring_points(x, y, arr, indirect=True):
    nbrs = neighboring_points(x, y, arr, indirect)
    sx, sy, _ = arr.shape
    return [sx * coord[0] + coord[1] for coord in nbrs]

@jit(nopython=True)
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

def show_image(img, scale=1.0, window_name='frame'):
    cv2.imshow(f'{window_name}', cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
    return cv2.waitKeyEx(0)

def show_images(img, scale=1.0, window_name='frame'):
    for i, j in enumerate(img):
        cv2.imshow(f'{window_name} {i}', cv2.resize(j, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))
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
    return cv2.imread(PNG_TMPL.format(DATA_DIR=DATA_DIR, game=game, play_number=play_number, frame_number=frame_number), cv2.IMREAD_UNCHANGED)
def get_playthrough(play_number, game='SuperMarioBros-Nes'):
    if play_number is not None:
        return dict(np.load(f'{PLAY_DIR}/{game}/{play_number}/frames.npz'))
    else:
        raise TypeError

def save_partially_processed_playthrough(raw, play_number, game, partial=None):
    if partial is not None:
        np.savez(f'{PLAY_DIR}/{game}/{play_number}/frames.npz', raw=raw, in_progress=partial)
    else:
        np.savez(f'{PLAY_DIR}/{game}/{play_number}/frames.npz', raw=raw)

def get_partially_processed_playthrough(array, play_number, game='SuperMarioBros-Nes'):
    return np.load(f'{PLAY_DIR}/{game}/{play_number}/frames.npz')['in_progress']

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

def add_rule(img):
    n_img = np.zeros((img.shape[0] + 1, img.shape[1] + 1, img.shape[2]))
    black = np.zeros((4), dtype=np.uint8)
    white = np.full((4), 255, dtype=np.uint8)
    column, row = np.array([black, white] * math.ceil(n_img.shape[0] / 2), dtype='uint8'), np.array([black, white] * math.ceil(n_img.shape[1] / 2), dtype='uint8')
    n_img[1:, 1:] = img
    n_img[0, :] = row
    n_img[:, 0] = column
    return n_img

def fill_all_patches(img, plist):
    for i in plist:
        img = i.fill_patch(img)
    return img

def get_palette(img):
    sx, sy, schan = img.shape
    img = img.reshape((sx * sy, schan))
    return list(set([tuple(pixel) for pixel in img]))

def migrate_play_through(play_through_data, play_number, game):
    if 'raw' not in play_through_data:
        print('migrating play through: schema')
        start = time.time()
        play_through_data['raw'] = play_through_data['arr_0']
        del play_through_data['arr_0']
        save_partially_processed_playthrough(play_through_data['raw'], play_number, game=game)
        print(f'finished in {time.time() - start}')

    if np.array_equal(play_through_data['raw'][0][0][0], np.array([88, 148, 248], dtype='uint8')):
        print('migrating play through: color encoding')
        start = time.time()
        play_through_data['raw'] = np.array([cv2.cvtColor(frame, cv2.COLOR_RGB2BGRA) for frame in play_through_data['raw']], dtype=np.uint8)
        save_partially_processed_playthrough(play_through_data['raw'], play_number, game=game)
        print(f'finished in {time.time() - start}')

    if play_through_data['raw'].shape[-1] == 3:
        print('migrating play through: adding alpha channel')
        start = time.time()
        play_through_data['raw'] = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA) for frame in play_through_data['raw']])
        save_partially_processed_playthrough(play_through_data['raw'], play_number, game=game)
        print(f'finished in {time.time() - start}')

    return play_through_data

def compare_images(img1, img2, sep=10, sep_color=(255, 255, 255), scale=1.0):
    return np.hstack((img1, np.array([[sep_color] * sep] * img1.shape[2]), img2))

def palettize_image(image, palette):
    p_index = {c:i for i, c in enumerate(palette)}
    palettized = np.zeros((image.shape[0], image.shape[1], 1), dtype='uint8')
    for x, row in enumerate(palettized):
        for y, pixel in enumerate(row):
            pixel = p_index[tuple(image[x][y])]
    return palettized

