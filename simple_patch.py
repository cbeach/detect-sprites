import math
from functools import partial
from glob import glob
import multiprocessing as mp
import sys
import time

import cv2
from numba import jit
import numpy as np
from scipy.spatial.ckdtree import cKDTree
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sprites.sprite_util import neighboring_points, get_playthrough, show_image, show_images

np.set_printoptions(threshold=10000000, linewidth=1000000000)

@jit(nopython=True)
def quick_parse(img: np.array, ntype: bool=False):
    if img.shape[-1] == 4:
        visited = (img[:, :, 3] == 0).astype(np.ubyte)
    else:
        visited = np.zeros(img.shape[:-1], dtype=np.ubyte)

    patches = [(0, 0, 0, 0, 0, 0, 0)]  # Give numba's type inference something to work with
    patches.pop()
    masks = [np.zeros((1, 1), dtype=np.ubyte)]  # numba hint
    masks.pop()

    for x in range(len(img)):
        for y in range(len(img[x])):
            if visited[x][y] == True:
                continue

            seed = (x, y)
            patch = [(x, y)]  # Hint for numba's type inference system
            stack = [patch.pop()]
            visited[x][y] = True
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

            #x_arr = np.array([c[0] for c in patch_arr])
            #y_arr = np.array([c[1] for c in patch_arr])
            assert(x_arr.shape[0] == patch_arr.shape[0])
            assert(y_arr.shape[0] == patch_arr.shape[0])
            for i in range(len(x_arr)):
                assert(patch_arr[i][0] == x_arr[i])
                assert(patch_arr[i][1] == y_arr[i])

            x1, y1 = min(x_arr), min(y_arr),
            w, h = (max(x_arr) + 1) - x1 , (max(y_arr) + 1) - y1
            nx_arr = x_arr - x1
            ny_arr = y_arr - y1
            mask = np.zeros(img.shape[:-1], dtype=np.ubyte)
            points = []
            for i in range(len(patch_arr)):
                nx = patch_arr[i][0] - x1
                ny = patch_arr[i][1] - y1
                points.append((nx, ny))
                mask[nx][ny] = 1
            masks.append(mask[:w, :h].copy())
            patches.append((seed[0], seed[1], len(patch), x1, y1, w, h))

    return np.array(patches, dtype=np.uint32)
def par(pt, img_count, start=0, pool=None):
    p_func = partial(quick_parse, ntype=False)
    if pool is None:
        pool = mp.Pool(8)
    #return np.array(pool.map(p_func, pt), dtype=np.int32)
    return pool.map(p_func, pt[start:start+img_count])
def ser(pt, img_count):
    temp = []
    for i, img in enumerate(pt[:img_count]):
        print(i)
        temp.append(quick_parse(img, False))
    return temp
def normalize_knn(kdt, features, k):
    if len(kdt.data) == 1:
        return np.zeros((1, k)), np.zeros((1, k, 2))
    dd, ii = kdt.query(features, k=k)
    mag = dd[:, 1:]
    all_coords = np.array([kdt.data[ii[i]] for i in range(ii.shape[0])])
    coords = all_coords[:, 1:]
    for c in coords:
        pass
    flat_coords = coords.reshape((coords.shape[0] * coords.shape[1], 2))
    print('f', flat_coords[:10])
    print('t', flat_coords[:10])
    sys.exit(0)
    unit_vect = preprocessing.normalize(flat_coords).reshape(coords.shape)
    return mag, unit_vect

def bench():
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    frame_number = 0
    ntype = False
    total_time = 0
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    sprite_clouds = [quick_parse(img, ntype=False) for img in sprites]
    sprite_kdt = [cKDTree(data=cloud[:, :2]) for cloud in sprite_clouds]

    mario_sprite, mario_cloud, mario_kdt = sprites[1], sprite_clouds[1], sprite_kdt[1]
    m_mag, m_uv = normalize_knn(sprite_kdt[1], mario_cloud[:, :2], k=4)

    pt = get_playthrough(play_number, game=game)['raw']
    img_count = 1
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    quick_parse(sprites[1], ntype=False)
    pool = mp.Pool(8)
    start = time.time()
    #features = [f[:, :2] for f in par(pt, img_count=img_count)]
    full_features = [f for f in par(pt, img_count=img_count, pool=pool)]
    features = [f[:, :2] for f in full_features]
    total_time += time.time() - start
    print('features', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    start = time.time()
    trees = pool.map(cKDTree, features)
    #trees = [ for f in features]
    total_time += time.time() - start
    print('trees', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    start = time.time()
    norm = pool.starmap(partial(normalize_knn, k=4), zip(trees, features))
    #[normalize_knn(kdt, f, k=4) for kdt, f in zip(trees, features)]
    total_time += time.time() - start
    print('normalization', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    start = time.time()
    indx = [index(full_features[i], n[0], n[1], m_mag, m_uv) for i, n in enumerate(norm)]
    total_time += time.time() - start
    print('indexing', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    print('total', total_time, 'average', float(total_time) / float(img_count))

    print(norm[0][0].shape, norm[0][1].shape)

def main():
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    frame_number = 0
    ntype = False
    total_time = 0
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    sprite_clouds = [quick_parse(img, ntype=False) for img in sprites]
    sprite_kdt = [cKDTree(data=cloud[:, :2]) for cloud in sprite_clouds]

    mario_sprite, mario_cloud, mario_kdt = sprites[1], sprite_clouds[1], sprite_kdt[1]
    m_mag, m_uv = normalize_knn(sprite_kdt[1], mario_cloud[:, :2], k=4)

    pt = get_playthrough(play_number, game=game)['raw']
    img_count = 1

    # Load test image
    test_img = cv2.imread('knn_test.png')

    start = time.time()

    # Parse frame
    full_features = quick_parse(test_img, ntype=False)
    features = full_features[:, :2]
    total_time += time.time() - start
    print('features', time.time() - start, 'average', float(time.time() - start) / float(img_count))

    # KDTrees
    start = time.time()
    tree = cKDTree(data=features)
    #trees = [ for f in features]
    total_time += time.time() - start
    print('trees', time.time() - start, 'average', float(time.time() - start) / float(img_count))

    # Normalize vectors
    start = time.time()
    norm = normalize_knn(tree, features, k=4)
    total_time += time.time() - start
    print('normalization', time.time() - start, 'average', float(time.time() - start) / float(img_count))

    # Match points
    start = time.time()


    rect = cv2.rectangle(test_img, (full_features[1][4], full_features[1][3]), (full_features[1][4] + full_features[1][6], full_features[1][3] + full_features[1][5]), (0, 255, 0))
    matches = [(i, f) for i, f in enumerate(mario_cloud) if f[2] == 8]
    rects = []
    for i, f in matches:
        print(i)
        rects.append(cv2.rectangle(mario_sprite.copy(), (f[4], f[3]), (f[4] + f[6], f[3] + f[5]), (0, 255, 0)))
    #show_image(rect, scale=3)
    print(m_mag[0])
    print(m_uv[0])
    show_images(rects, scale=16)
    return

    print('target', full_features[1])
    rnorm = (np.round(norm[0], decimals=5), np.round(norm[1], decimals=5))
    match_features(full_features[1], (rnorm[0][1], rnorm[1][1]), mario_cloud, (np.round(m_mag, decimals=5), np.round(m_uv, decimals=5)))
    #indx = [index(full_features, norm[0], norm[1], mario_cloud, m_mag, m_uv) for i, n in enumerate(norm)]
    total_time += time.time() - start
    print('indexing', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    print('total', total_time, 'average', float(total_time) / float(img_count))

    print(norm[0][0].shape, norm[0][1].shape)

def match_features(tfeat, tnbrs, sfeats, snbrs):
    target_patch_area = tfeat[2]
    sprite_patch_area = [i[2] for i in sfeats]
    area_matches = [i for i, area in enumerate(sprite_patch_area) if area == target_patch_area]
    print('area_matches', area_matches)
    sprite_mags = [snbrs[0][i] for i in area_matches]
    sprite_uvs = [snbrs[1][i] for i in area_matches]
    print('tmag')
    for t in tnbrs[0]:
        print(t)
    print(tnbrs[0][0], tnbrs[0][1], tnbrs[0][2])
    print(tnbrs[1][0], tnbrs[1][1], tnbrs[1][2])
    #print('sprite_mags', sprite_mags)
    print()
    print('sprite_mags')
    print(0, sprite_mags[0], sprite_uvs[0][0], sprite_uvs[0][1])
    print(1, sprite_mags[1], sprite_uvs[1][0], sprite_uvs[1][1])
    print(2, sprite_mags[2], sprite_uvs[2][1])
    t_mag = tnbrs[0]
    t_uv = tnbrs[1]
    for i, sm in enumerate(sprite_mags):
        for j, t in enumerate(t_mag):
            if np.array_equal(sm, t):
                print(i, j)
                print('s', sm)
                print('t', t)



def index(features, mag, uv, sfeatures, smag, suv, precision=3):
    ms, uvs = mag.shape, uv.shape
    rmag, ruv = np.round(mag, decimals=precision), np.round(uv, decimals=precision)
    rsmag, rsuv = np.round(smag, decimals=precision), np.round(suv, decimals=precision)
    target_patch_area = [i[2] for i in features]
    print('target features len', len(features))
    sprite_patch_area = [i[2] for i in sfeatures]
    area_match = [[i for i, target_area in enumerate(target_patch_area) if target_area == sprite_area] for sprite_area in sprite_patch_area]
    for i, a in enumerate(area_match):
        print('area_match', i, a)
    #for i in sfeatures:
    return

# features: [(first_pixel_x, first_pixel_y, area, bounding_box_x1, bounding_box_y1, w, h)]
if __name__ == '__main__':
    main()
