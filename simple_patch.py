from functools import partial
from glob import glob
import multiprocessing as mp
import sys
import time

import cv2
from numba import jit
import numpy as np
from scipy.spatial.ckdtree import cKDTree

from sprites.sprite_util import neighboring_points, get_playthrough, show_image

@jit(nopython=True)
def quick_parse(img: np.array, ntype: bool=False):
    visited = np.zeros(img.shape[:-1], dtype=np.bool8)
    patches = [(0, 0, 0, 0, 0, 0, 0)]  # Give numba's type inference something to work with
    patches.pop()

    for x in range(len(img)):
        for y in range(len(img[x])):
            if visited[x][y] == True:
                continue
            seed = (x, y)
            patch = [(x, y)]  # Hint for numba's type inference system
            stack = [patch.pop()]
            visited[x][y] = 0
            while len(stack) > 0:
                current_pixel = stack.pop()
                patch.append(current_pixel)
                nbr_coords = neighboring_points(current_pixel[0], current_pixel[1], img, ntype)

                for i, j in nbr_coords:
                    if visited[i][j] == False and np.array_equal(img[i][j], img[x][y]):
                        stack.append((i, j))
                        visited[i][j] = True

            patch_arr = np.array(patch, dtype=np.ubyte)
            x_arr = patch_arr[:, 0]
            y_arr = patch_arr[:, 1]
            x1, y1 = min(x_arr), min(y_arr),
            w, h = (max(x_arr) + 1) - x1 , (max(y_arr) + 1) - y1
            patches.append((seed[0], seed[1], len(patch), x1, y1, w, h))

    return np.array(patches, dtype=np.uint32)

def par(pt, img_count):
    p_func = partial(quick_parse, ntype=False)
    pool = mp.Pool(8)
    #return np.array(pool.map(p_func, pt), dtype=np.int32)
    return pool.map(p_func, pt[:img_count])


def ser(pt, img_count):
    temp = []
    for i, img in enumerate(pt[:img_count]):
        print(i)
        temp.append(quick_parse(img, False))
    return temp

if __name__ == '__main__':
    #np.set_printoptions(linewidth=1000000)
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    #game = 'QBert-Nes'
    #play_number = 3928
    frame_number = 256
    ntype = False

    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    sprite_clouds = [quick_parse(img, ntype=False) for img in sprites]

    pt = get_playthrough(play_number, game=game)['raw']
    img_count = len(pt)
    start = time.time()
    print(img_count)
    features = par(pt, img_count)
    print(f'parse duration: {time.time() - start}\n      average: {float(time.time() - start) / float(img_count)}')
    start = time.time()
    print(pt[0].shape, pt[0].shape[0] * pt[1].shape[1])
    print('\nfeatures')

    feature = features[0][:, :2]
    kdt = cKDTree(data=feature)
    print(f'kdt duration: {time.time() - start}\n    average: {float(time.time() - start) / float(img_count)}')

    dd, ii = kdt.query(feature, k=4, distance_upper_bound=50, n_jobs=8)
    print('ii', ii[0])
    coords = np.array([kdt.data[ii[i]] for i in range(ii.shape[0])])
    this, that = coords[:, 0], coords[:, 1:]
    the_other = np.zeros_like(that)
    nl = '\n'
    for i in range(len(this)):
        the_other[i] = that[i] - this[i]
        if i < 10:
            print(f'{i}')
            print(f'this: {str(this[i]).replace(nl, ", ")}\nthat: {str(that[i]).replace(nl, ", ")}')
            print(f'translated: {str(that[i] - this[i]).replace(nl, ", ")}\n')

    print(the_other[:10])
    #print(dd[:10, 1:])
    #print(ii[:10, 1:])
