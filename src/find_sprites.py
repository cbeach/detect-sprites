import itertools
from collections import defaultdict
import json
from glob import glob
import numpy as np
import sys
from pathlib import Path, PurePath
import random
import progressbar
from multiprocessing import Pool
import time


import cv2
from pyspark import SparkContext, SparkConf

#for i, s in enumerate(sprites):
#    cv2.imshow('sprinte', s)
#    cv2.waitKey(1)
threshold = 0.99
mask_color = np.array([143, 39, 146])

def load_sprites():
    sprite_paths = list(glob('sprites/SuperMarioBros-Nes/*.png'))
    #sprite_paths = ['./sprites/SuperMarioBros-Nes/lil_mario_2.png']
    #sprite_paths = ['./sprites/SuperMarioBros-Nes/cloud_1.png', './sprites/SuperMarioBros-Nes/Zero.png', './sprites/SuperMarioBros-Nes/cloud_2.png', 'sprites/SuperMarioBros-Nes/lil_mario_2.png']

    # sort by size
    sprites = [cv2.imread(i) for i in sprite_paths]


    ss = zip(sprite_paths, sprites)

    def img_size(img):
        s = img[1].shape
        return s[0] * s[1]

    ss = sorted(ss, key=img_size, reverse=True)
    sprite_paths = [i[0] for i in ss]
    sprites = [i[1] for i in ss]

    sprite_paths.extend(sprite_paths)

    sprites.extend([cv2.flip(j, 1) for j in [np.copy(i) for i in sprites]])

    masks = [np.copy(i) for i in sprites]
    stats = [0] * len(masks)

    for i, s in enumerate(sprites):
        for j, row in enumerate(s):
            for k, pixel in enumerate(row):
                if np.array_equal(mask_color, pixel):
                    masks[i][j][k][0] = 0
                    masks[i][j][k][1] = 0
                    masks[i][j][k][2] = 0
                else:
                    masks[i][j][k][0] = 255
                    masks[i][j][k][1] = 255
                    masks[i][j][k][2] = 255

    return sprite_paths, sprites, masks, stats

def mask_frame(frame, mask, x, y, w, h):
    for i in range(w):
        for j in range(h):
            if np.array_equal(mask[i][j], np.array([255, 255, 255])):
                frame[y + i][x + j] = mask_color
    return frame

def find_sprite(fsm):
    frame_path, sprites, masks = fsm
    #for i, frame_path in enumerate(frame_paths):
    frame = cv2.imread(frame_path)
    frame_copy = np.copy(frame)
    frame_path.split
    pure_path = PurePath(frame_path)
    basename = pure_path.name
    play_number, frame_number = basename.split('.')[0].split('-')

    all_max_vals = []
    all_max_locs = []
    for j, sprite in enumerate(sprites):
        res = cv2.matchTemplate(frame, sprite, cv2.TM_CCORR_NORMED, None, masks[j])
        matches = np.where(res >= threshold)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res, None)

        w, h = sprite.shape[:-1]
        print(max_val, max_loc, (h, w))
        if max_val >= threshold:
            for pt in zip(*matches[::-1]):
                frame = mask_frame(frame, masks[j], pt[0], pt[1], w, h)
                cv2.rectangle(frame_copy, pt, (pt[0] + h, pt[1] + w), (0,0,255), 1)
        # TODO: record the sprite locations
        all_max_vals.append(max_val)
        all_max_locs.append(max_loc)

    cv2.imwrite('res.png', frame_copy)
    cv2.imwrite('mres.png', frame)

    max_val = max(all_max_vals)
    sprite_index = all_max_vals.index(max_val)
    max_loc = all_max_locs[sprite_index]
    pos_sprite = sprites[sprite_index]
    pos_mask = masks[sprite_index]
    w, h = sprites[sprite_index].shape[:-1]
    x, y = max_loc


    sprite_loc = {
        'play': play_number,
        'frame': frame_number,
    }

    if max_val >= threshold:
        sprite_loc['loc'] = ((x, y), (x + h, y + w))
        sprite_loc['ind'] = sprite_index

    return sprite_loc, sprite_index

def chunks(l, n):
    """Yield successive n-sized groups of images from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def main():
    print('Loading sprites')
    sprite_paths, sprites, masks, stats = load_sprites()

    print('Loading image file list')
    with open('./image_files', 'r') as fp:
        frame_paths = [i.strip() for i in fp.readlines()]
    random.shuffle(frame_paths)

    print('Loading sprite locations from json')
    try:
        with open('sprite_locations.json', 'r') as fp:
            all_locations = json.load(fp)
    except FileNotFoundError:
        all_locations = []
    except json.decoder.JSONDecodeError:
        all_locations = []

    print('filtering frames')


    filtered_frame_paths = frame_paths
    if len(all_locations) > 0:
        frame_map = defaultdict(dict)
        def play_and_frame(path):
            path.split
            pure_path = PurePath(path)
            basename = pure_path.name
            play_number, frame_number = basename.split('.')[0].split('-')
            return not frame_map[play_number].get(frame_number, False)

        for i in all_locations:
            frame_map[i['play']][i['frame']] = True

        filtered_frame_paths = list(filter(play_and_frame, frame_paths))

    total_frames = 0
    sample_size = 100 # Set to None to process all of the images
    #print(find_sprite(('4285-59.png', sprites, masks, sprite_paths)))

    print('finding sprites...')
    with Pool(8) as p:
        for i in list(chunks(filtered_frame_paths[:sample_size], 128)):
            fps = [(f, sprites, masks) for f in i]
            res = p.map(find_sprite, fps)
            for sprite_loc, sprite_index in res:
                stats[sprite_index] += 1
                if 'ind' in sprite_loc:
                    sprite_loc['sprite'] = sprite_paths[sprite_loc['ind']]
                    del sprite_loc['ind']
                all_locations.append(sprite_loc)
                total_frames += 1

            with open('sprite_locations.json', 'w') as fp:
                json.dump(all_locations, fp, indent=2)

            stats, masks, sprites, sprite_paths = zip(*sorted(zip(stats, masks, sprites, sprite_paths), reverse=True, key=lambda x: x[0]))
            stats = list(stats)


def get_patches(frame):
    mask = np.ones(frame.shape[:-1])
    bg_color = frame[0][0].copy()
    size = frame.shape[0] * frame.shape[1]
    counter = 0.0
    patches = []
    hat = (216, 40, 0)
    for i, row in enumerate(mask):
        for j, pix in enumerate(row):
            counter += 1
            if pix == 1:
                if np.array_equal(bg_color, frame[i][j]) is True:
                    mask[i][j] = 0
                    continue
                patches.append((frame[i][j], iterate_on_patch(frame, mask, i, j)))
                if (185, 88) in patches[-1][1]:
                    print(len(patches) - 1, patches[ - 1])
    return patches


def iterate_on_patch(frame, mask, x, y):
    stack = [(x, y)]
    mask[x][y] = 0
    patch = []
    while len(stack) > 0:
        cur = stack.pop()
        patch.append(cur)
        for i, j in get_neighbors(frame, *cur):
            if mask[i][j] == 1:
                stack.append((i, j))
                mask[i][j] = 0
    return patch

def get_neighbors(frame, x, y):
    neighbors = []
    if x > 0 and y > 0 and np.array_equal(frame[x-1][y-1], frame[x][y]):
        neighbors.append((x-1, y-1))
    if y > 0 and np.array_equal(frame[x][y-1], frame[x][y]):
        neighbors.append((x, y-1))
    if x < frame.shape[0] - 1 and y > 0 and np.array_equal(frame[x+1][y-1], frame[x][y]):
        neighbors.append((x+1, y-1))

    if x > 0 and np.array_equal(frame[x-1][y], frame[x][y]):
        neighbors.append((x-1, y))
    #if x > 0 and np.array_equal(frame[x][y], frame[x][y]):
    #    neighbors.append((x, y))
    if x < frame.shape[0] - 1 and np.array_equal(frame[x+1][y], frame[x][y]):
        neighbors.append((x+1, y))

    if x > 0 and y < frame.shape[1] - 1 and np.array_equal(frame[x-1][y+1], frame[x][y]):
        neighbors.append((x-1, y+1))
    if y < frame.shape[1] - 1 and np.array_equal(frame[x][y+1], frame[x][y]):
        neighbors.append((x, y+1))
    if x < frame.shape[0] - 1 and y < frame.shape[1] - 1 and np.array_equal(frame[x+1][y+1], frame[x][y]):
        neighbors.append((x+1, y+1))

    return neighbors


def bounding_box(patch):
    x = [i[0] for i in patch]
    y = [i[1] for i in patch]
    return ((min(x), min(y)), (max(x) + 1, max(y) + 1))

def draw_bounding_box(img, bb):
    cp = img.copy()
    cv2.rectangle(cp, (bb[0][1], bb[0][0]), (bb[1][1], bb[1][0]), (0,255,0),1)
    return cp

def fill_patch(img, patch):
    cp = img.copy()
    for x, y in patch[1]:
        cp[x][y][0] = 0
        cp[x][y][1] = 255
        cp[x][y][2] = 0
    return cp


if __name__ == "__main__":
    img = cv2.imread('1008-693.png')
    #main()
    patches = get_patches(img)
    palette = set([tuple(p[0]) for p in patches])
    print(len(palette), palette)
    bbs = []
    bb_sizes = []
    bb_dims = []
    for c, p in patches:
        bb = bounding_box(p)
        bbs.append(bb)
        bb_sizes.append((bb[1][0] - bb[0][0]) * (bb[1][1] - bb[0][1]))
        bb_dims.append(((bb[1][0] - bb[0][0]), (bb[1][1] - bb[0][1])))


    minbb = min(bb_sizes)
    maxbb = max(bb_sizes)
    print("min", minbb, bb_dims[bb_sizes.index(minbb)], bbs[bb_sizes.index(minbb)])
    print("max", maxbb, bb_dims[bb_sizes.index(maxbb)], bbs[bb_sizes.index(maxbb)])
    print(patches[bb_sizes.index(maxbb)][0])

    bb_drawn = draw_bounding_box(img, bbs[bb_sizes.index(maxbb)])
    pfilled = fill_patch(img, patches[bb_sizes.index(maxbb)])
    scaled = cv2.resize(pfilled, (pfilled.shape[1]*3, pfilled.shape[0]*3))
    cv2.imshow('frame', scaled)
    cv2.waitKey(0)
