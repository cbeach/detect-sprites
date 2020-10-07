from collections import namedtuple
import math
from functools import partial
from glob import glob
import multiprocessing as mp
import os
import random
import sys
import time

import cv2
from numba import jit
import numpy as np
from recordclass import RecordClass
from scipy.spatial.ckdtree import cKDTree
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error

from sprites.sprite_util import neighboring_points, get_playthrough, show_image, show_images
from sprites.find import fit_bounding_box
from sprites.patch_graph import FrameGraph

# features: [(first_pixel_x, first_pixel_y, area, bounding_box_x1, bounding_box_y1, w, h)]
FeatureSignature = namedtuple('FeatureSignature', ['first_x', 'first_y', 'area', 'bbx1', 'bby1', 'w', 'h', 'mag', 'uv'])

np.set_printoptions(threshold=10000000, linewidth=1000000000)
shrink = True

@jit(nopython=True)
def patch_hash(patch: np.array):
    sx, sy = list(patch.shape[:2])
    area = sx * sy
    patch_hash = np.zeros((math.ceil(float(area + 16) / 64.0)), dtype=np.int64)

    sx_hash = np.zeros((8), dtype=np.int64)
    for i in range(8):
        sx_hash[i] = 1 if 0b10000000 & sx else 0
        sx = sx << 1

    sy_hash = np.zeros((8), dtype=np.int64)
    for i in range(8):
        sy_hash[i] = 1 if 0b10000000 & sy else 0
        sy = sy << 1

    p64 = np.concatenate((patch.astype(np.int64).flatten(), sx_hash, sy_hash))
    patch_hash_index = 0
    mini_hash = 1
    first_iteration = True
    for i, pix in enumerate(p64.flatten()):
        mini_hash = mini_hash << 1
        if pix:
            mini_hash += 1

        if first_iteration is True and i == 62:
            patch_hash[patch_hash_index] = mini_hash
            mini_hash = 0
            patch_hash_index = i // 64
            first_iteration = False
        elif patch_hash_index != i // 64:
            patch_hash[patch_hash_index] = mini_hash
            mini_hash = 0
            patch_hash_index = i // 64

    patch_hash[patch_hash_index] = mini_hash

    return patch_hash
@jit(nopython=True)
def quick_parse(img: np.array, ntype: bool, shrink_mask: bool = True):
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
                if shrink_mask is True:
                    mask[nx][ny] = 1
                else:
                    mask[patch_arr[i][0]][patch_arr[i][1]] = 1
            if shrink_mask is True:
                mask = mask[:w, :h].copy()
            else:
                mask = mask.copy()
            patch_hash(mask)
            masks.append(mask)
            patches.append((seed[0], seed[1], len(patch), x1, y1, w, h))

    return np.array(patches, dtype=np.uint32), masks
def par(pt, img_count, ntype, start=0, pool=None):
    p_func = partial(quick_parse, ntype=ntype, shrink_mask=True)
    if pool is None:
        pool = mp.Pool(8)
    return pool.map(p_func, pt[start:start+img_count])
def ser(pt, img_count, ntype, start=0):
    patches = []
    masks = []
    for i, img in enumerate(pt[start:start+img_count]):
        print(i)
        p, m = quick_parse(img, False)
        patches.append(p)
        masks.append(m)
    return patches, masks
def normalize_knn(kdt, features, k, translate=True):
    if len(kdt.data) == 1:
        return np.zeros((1, k)), np.zeros((1, k, 2))
    dd, ii = kdt.query(features, k=k)
    mag = dd[:, 1:]
    all_coords = np.array([kdt.data[ii[i]] for i in range(ii.shape[0])])
    origins = all_coords[:, 0]
    coords = all_coords[:, 1:]
    for i, c in enumerate(coords):
        coords[i] = c - origins[i]
    flat_coords = coords.reshape((coords.shape[0] * coords.shape[1], 2))
    unit_vect = preprocessing.normalize(flat_coords).reshape(coords.shape)
    return mag, unit_vect
def color_mask(img, mask, color=(0, 255, 0)):
    c = np.array(color, dtype=np.ubyte)
    img = img.copy()
    for x, y in zip(*np.nonzero(mask)):
        img[x][y] = c
    return img
def populate_features(feats, mags, uvs):
    return [FeatureSignature(first_x=feats[i][0], first_y=feats[i][1], area=feats[i][2], bbx1=feats[i][3], bby1=feats[i][4],
                             w=feats[i][5], h=feats[i][6], mag=mags[i], uv=uvs[i]) for i, _ in enumerate(feats)]
def bench():
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    ntype = False
    pool = mp.Pool(8)
    total_time = 0
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    quick_parse(sprites[0], ntype=ntype, shrink_mask=shrink)
    parsed_sprites = [f for f in par(sprites, img_count=len(sprites), ntype=ntype, pool=pool)]
    sprite_full_features = np.array([i[0] for i in parsed_sprites])
    sprite_masks = [i[1] for i in parsed_sprites]
    #parsed_sprites = [quick_parse(img, ntype=ntype, shrink_mask=shrink) for img in sprites]
    sprite_kdt = [cKDTree(data=cloud[:, :2]) for cloud in sprite_full_features]

    snorm = pool.starmap(partial(normalize_knn, k=8), zip(sprite_kdt, [sc[:, :2] for sc in sprite_full_features]))
    smags, suvs = [i[0] for i in snorm], [i[1] for i in snorm]
    sfeatures = []
    for i in range(len(sprites)):
        sfeatures.append(populate_features(sprite_full_features[i], smags[i], suvs[i]))

    mario_sprite, mario_sprite_point_cloud, mario_kdt = sprites[1], sfeatures[1], sprite_kdt[1]
    mmag, muv = normalize_knn(sprite_kdt[1], [i[:2] for i in mario_sprite_point_cloud], k=8)

    pt = get_playthrough(play_number, game=game)['raw']
    img_count = len(pt)
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    # ensure quick_parse function is compiled
    quick_parse(sprites[1], ntype=False)
    start = time.time()
    #features = [f[:, :2] for f in par(pt, img_count=img_count)]
    parsed_frames = [f for f in par(pt, img_count=img_count, ntype=ntype, pool=pool)]
    full_features = [i[0] for i in parsed_frames]
    masks = [i[1] for i in parsed_frames]
    features = [f[:, :2] for f in full_features]
    total_time += time.time() - start
    print('features', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    start = time.time()
    trees = pool.map(cKDTree, features)
    #trees = [ for f in features]
    total_time += time.time() - start
    print('trees', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    start = time.time()
    norm = pool.starmap(partial(normalize_knn, k=12), zip(trees, features))
    tmag, tuv = norm[0]
    #[normalize_knn(kdt, f, k=12) for kdt, f in zip(trees, features)]
    total_time += time.time() - start
    print('normalization', time.time() - start, 'average', float(time.time() - start) / float(img_count))
    start = time.time()

    total_time += time.time() - start
    print('matching', time.time() - start, 'average', float(time.time() - start) / float(img_count))

    #for i in range(len(sprites)):
    #    for j, f in enumerate(full_features):
    #        #print('len sf', len(sfeatures[i]))
    #        #print('len sm', len(smags[i]))
    #        #print('len su', len(suvs[i]))

    #        match_all_featues(full_features[j], tmag[j], tuv[j], sfeatures[i], smags[i], suvs[i])
    print('total', total_time, 'average', float(total_time) / float(img_count))

def main():
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    frame_number = 0
    ntype = False
    total_time = 0
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    parsed_sprites = [quick_parse(img, ntype=ntype, shrink_mask=shrink) for img in sprites]
    sprite_clouds = [i[0] for i in parsed_sprites]
    sprite_masks = [i[1] for i in parsed_sprites]
    sprite_kdt = [cKDTree(data=cloud[:, :2]) for cloud in sprite_clouds]


    mario_sprite, mario_sprite_point_cloud, mario_kdt = sprites[1], sprite_clouds[1], sprite_kdt[1]
    mmag, muv = normalize_knn(sprite_kdt[1], mario_sprite_point_cloud[:, :2], k=8)

    pt = get_playthrough(play_number, game=game)['raw']
    img_count = 1

    # Load test image
    test_img = cv2.imread('knn_test.png')

    img1, s, bb = generate_random_test_image(test_img.shape, sprites, 5, overlap=False, bg_color=(0, 0, 0))
    show_image(img1, scale=3)

    start = time.time()

    # Parse frame
    full_features, masks = quick_parse(test_img, ntype=ntype, shrink_mask=shrink)
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
    tmag, tuv = normalize_knn(tree, features, k=12)
    total_time += time.time() - start
    print('normalization', time.time() - start, 'average', float(time.time() - start) / float(img_count))

    # Match points
    start = time.time()
    matches = []
    smatches = np.zeros((len(mario_sprite_point_cloud)), dtype=np.bool8)
    for i, tf in enumerate(full_features):
        #print([match_features(full_features[i], tmag[i], tuv[i], f, m, u) for f, m, u in zip(mario_cloud, mmag, muv)])
        matches.append([match_feature(full_features[i], tmag[i], tuv[i], f, m, u, p=True) for f, m, u in zip(mario_sprite_point_cloud, mmag, muv)])
        try:
            smatches[matches[-1].index(True)] = True
        except ValueError:
            pass
        #print([int(j) for j in matches[-1]], sum([int(j) for j in matches[-1]]), i)

    print(smatches)
    temp = []
    with np.printoptions(threshold=10000000, linewidth=1000000000, formatter={'bool': lambda b: '1,' if b else ' ,'}):
        for i, m in enumerate(matches):
            print(f'{i: <3}: {np.array(m, dtype=np.bool8)}')
    for i in matches:
        try:
            temp.append(i.index(True))
        except ValueError:
            temp.append(-1)
    print(temp)
def generate_random_test_image(shape, sprites, num_sprites, overlap=False, bg_color=(248, 148, 88)):
    img = np.zeros(shape, dtype=np.ubyte)
    for x, row in enumerate(img):
        for y, _ in enumerate(row):
            img[x][y][0] = bg_color[0]
            img[x][y][1] = bg_color[1]
            img[x][y][2] = bg_color[2]

    overlaps = lambda bb1, bb2: (bb1[1][0] >= bb2[0][0] and bb2[1][0] >= bb1[0][0]) and (bb1[1][1] >= bb2[0][1] and bb2[1][1] >= bb1[0][1])
    rsprites = []
    bbs = []
    for i in range(num_sprites):
        rsprite_i = random.randrange(len(sprites))
        rsprite = sprites[rsprite_i]
        added = False
        for j in range(100):
            x, y = random.randrange(shape[0]), random.randrange(shape[1])
            xs, ys = rsprite.shape[:2]
            bb1 = ((x, y), (x + xs, y + ys))
            if len(bbs) == 0 or not any([overlaps(bb1, bb2) for bb2 in bbs]):
                rsprites.append(rsprite_i)
                bbs.append(bb1)
                added = True
                break
        if added is False:
            raise RuntimeError('Too many attempts to add sprite to test image. try allowing overlap, orreducing the number of sprites')

    for s, bb in zip(rsprites, bbs):
        x1, y1, x2, y2 = fit_bounding_box(img, bb)
        sprite = sprites[s].copy()
        for x, row in enumerate(sprite):
            for y, _ in enumerate(row):
                if sprite[x][y][3] == 0:
                    sprite[x][y][0] = bg_color[0]
                    sprite[x][y][1] = bg_color[1]
                    sprite[x][y][2] = bg_color[2]
        img[x1:x2, y1:y2] = sprite[:x2 - x1, :y2 - y1, :3]

    return img, rsprites, bbs
#@jit(nopython=True)
def match_feature(tfeat, tmag, tuv, sfeat, smag, suv, p=False):
    if tfeat[2] != sfeat[2] or tfeat[5] != sfeat[5] or tfeat[6] != sfeat[6]:
        return False
    removed = np.zeros(smag.shape, dtype=np.bool8)
    matches = np.zeros(suv.shape[:1], dtype=np.bool8)
    for i, mag, uv in zip(range(len(tuv)), tmag, tuv):
        for j, sm, su in zip(range(len(suv)), smag, suv):
            if removed[j] == True:
                continue
            if np.array_equiv(uv, suv[j]) and np.array_equiv(mag, smag[j]):
                matches[j] = True
                removed[j] = True
                break
    if np.all(matches) != True:
        return False
    return True

def match_all_featues(tfeatures, tmag, tuv, sfeatures, smag, suv):
    f_matrix = []

    for i, tf in enumerate(tfeatures):
        s_matrix = []
        for j, sf in enumerate(sfeatures):
            #print('tf', tf)
            #print('sf', sf)
            #print()
            temp = []
            for k, _ in enumerate(smag[j]):
                #print(i, j, k)
                print('len(sf), len(smag[j])', len(sf), len(smag[j]))
                print('len(tf), len(tmag[i])', len(tf), len(tmag), tmag)
                #print(smag[j])
                a = tf
                b = tmag[i]
                c = tuv[i]
                d = sf
                e = smag[j][k]
                f = suv[j][k]
                temp.append(match_feature(a, b, c, d, e, f))
            s_matrix.append(temp)
        f_matrix.append(s_matrix)


def test_new_hashing_function():
    PROJECT_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    from sprites.db.data_store import DataStore
    ds = DataStore(f'{PROJECT_ROOT}/sprites/db/sqlite.db', games_path=f'{PROJECT_ROOT}/sprites/games.json', echo=False)
    r = np.array([0, 0, 255], dtype=np.uint8)
    g = np.array([0, 255, 0], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    c = np.array([255, 255, 0], dtype=np.uint8)
    m = np.array([255, 0, 255], dtype=np.uint8)

    w = np.array([255, 255, 255], dtype=np.uint8)
    a = np.array([128, 128, 128], dtype=np.uint8)

    ti1 = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
    ])

    tg1 = FrameGraph(ti1, bg_color=np.array([0,0,0]), ds=ds)
    assert(len(tg1.patches) == 10)

    ti2 = np.array([
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
    ])
    tg2 = FrameGraph(ti2, bg_color=np.array([0,0,0]), ds=ds)
    assert(len(tg2.patches) == 10)

    ti3 = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, w, w, w, w, c, c, c],
        [m, m, m, w, a, a, w, m, m, m],
        [r, r, r, w, a, a, w, r, r, r],
        [g, g, g, w, w, w, w, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
    ])
    tg3 = FrameGraph(ti3, bg_color=np.array([0,0,0]), ds=ds)
    assert(len(tg3.patches) == 16)

    ti4 = np.array([
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
    ])
    tg4 = FrameGraph(ti4, bg_color=w, ds=ds)
    tg4b = FrameGraph(ti4, ds=ds)
    sg4 = tg4.subgraphs()
    assert(len(tg4.patches) == 4)
    assert(len(sg4) == 2)

    ti5 = np.array([
        [r, r, w, r, r],
        [r, r, w, r, r],
        [b, b, w, b, b],
        [b, b, w, b, b],
    ])
    tg5 = FrameGraph(ti5, bg_color=w, ds=ds)
    sg5 = tg5.subgraphs()
    assert(len(tg5.patches) == 4)
    assert(len(sg5) == 2)
    print('x', tg4b.raw_frame.shape[0], bin(tg4b.raw_frame.shape[0]))
    print('y', tg4b.raw_frame.shape[1], bin(tg4b.raw_frame.shape[1]))
    print()

    def print_legacy_hash(node):
        h = bin(hash(node))
        print()
        print('  --- legacy ---')
        print('  hash', hash(node))
        print('  hash >> 16', hash(node) >> 16)
        print('  x', node.patch.shape()[0], bin(node.patch.shape()[0]))
        print('  y', node.patch.shape()[1], bin(node.patch.shape()[1]))
        print('  ', len(h) - 3, h)
        print('  ', len(h[3:-16]), h[3:-16])
        print('  ', len(h[-16:-8]), h[-16:-8])
        print('  ', len(h[-8:]), h[-8:])

    def print_new_hash(node):
        raw_patch = node.patch._patch._patch.astype(np.int64)
        print()
        print('  --- new ---')
        print('  raw_patch', raw_patch)
        print('  raw_patch.dtype', raw_patch.dtype)
        phash = patch_hash(raw_patch)
        h = bin(phash[0])
        print('  hash', phash[0])
        print('  x', node.patch.shape()[0], bin(node.patch.shape()[0]))
        print('  y', node.patch.shape()[1], bin(node.patch.shape()[1]))
        print('  ', len(h) - 3, h)
        print('  ', len(h[3:-16]), h[3:-16])
        print('  ', len(h[-16:-8]), h[-16:-8])
        print('  ', len(h[-8:]), h[-8:])


    for i, node in enumerate(tg4b.patches):
        print_legacy_hash(node)
        print_new_hash(node)

# features: [(first_pixel_x, first_pixel_y, area, bounding_box_x1, bounding_box_y1, w, h)]
if __name__ == '__main__':
    #test_new_hashing_function()
    bench()
    #main()
