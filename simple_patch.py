from collections import namedtuple, defaultdict
import math
from functools import partial, lru_cache
from itertools import product, groupby
import multiprocessing as mp
import os
from pprint import pprint
import random
import sys
import time

from matplotlib import pyplot
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
Patch = namedtuple('Patch', ['hash', 'mask', 'patch_index', 'first_x', 'first_y', 'area', 'bbx1', 'bby1', 'w', 'h'])
PatchVector = namedtuple('PatchVector', ['src_patch_hash', 'unit_vect', 'mag', 'dst_patch_hash', 'are_neighbors'])
Point = namedtuple('Point', ['x', 'y'])

np.set_printoptions(threshold=10000000, linewidth=1000000000)
shrink = True

def algo_v1():
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    ntype = False
    # parse
    start_time = time.time()
    playthrough_features, playthrough_masks, playthrough_hashes, playthrough_pix_to_patch_index \
        = parse_and_hash_playthrough(game, play_number, ntype, img_count=10)
    print('1', time.time() - start_time)
    start_time = time.time()
    playthrough_patches = encode_playthrough_patches(playthrough_features, playthrough_masks, playthrough_hashes)
    print('2', time.time() - start_time)
    start_time = time.time()
    playthrough_kdtrees = populate_kdtrees(playthrough_patches)
    print('3', time.time() - start_time)
    start_time = time.time()
    playthrough_neighbor_index = get_playthrough_neighbors_index(playthrough_pix_to_patch_index, ntype)
    print('4', time.time() - start_time)
    start_time = time.time()
    playthrough_vectors = encode_playthrough_vectors(
        playthrough_patches,
        playthrough_pix_to_patch_index,
        playthrough_kdtrees,
        playthrough_neighbor_index)
    print('5', time.time() - start_time)
    start_time = time.time()
    aggregated_vectors = aggregate_playthrough_vectors(playthrough_vectors)
    print('6', time.time() - start_time); start_time = time.time()
    #print(len(aggregated_vectors))
    #values = sorted(list(aggregated_vectors.values()))
    #magnitudes = sorted([i.mag for i in aggregated_vectors.keys()])
    #values = sorted(list(set(magnitudes)))
    #max_val = len(magnitudes)
    #distribution = np.zeros((len(values),), dtype=np.int32)
    #for i, group in enumerate(groupby(magnitudes)):
    #    distribution[i] = len(list(group[1]))
    #print(distribution.shape)
    #print(distribution)
    #pyplot.plot(values, distribution)
    #pyplot.show()
    filtered_playthrough_vectors = filter_playthrough_vectors(aggregated_vectors, playthrough_vectors)
    print('7', time.time() - start_time); start_time = time.time()
    filtered_aggregated_vectors = filter_aggregated_vectors(aggregated_vectors)
    print('8', time.time() - start_time); start_time = time.time()
    partition_vectors(filtered_aggregated_vectors, filtered_playthrough_vectors)
    print('9', time.time() - start_time); start_time = time.time()

def partition_vectors(aggregated_vectors, filtered_playthrough_vectors):
    filtered_aggregated_vectors = filter_aggregated_vectors(aggregated_vectors)
    count = 0
    for i, playthrough_vectors in enumerate(filtered_playthrough_vectors):
        sorted_vectors = sorted(playthrough_vectors, key=lambda v: filtered_aggregated_vectors[hashable_vector(v)])
        V = []
        for j, vector in enumerate(sorted_vectors):
            V.append(vector)
            matching_src_vectors = list(filter(lambda v: np.array_equal(v[0], vector[0]), sorted_vectors[j + 1:]))
            if len(matching_src_vectors) == 0:
                continue
            count += 1
            V.append(matching_src_vectors.pop())

    print(len(filtered_playthrough_vectors))


def filter_aggregated_vectors(aggregated_vectors):
    def filter_func(vector):
        return aggregated_vectors[vector] > 2 and vector.are_neighbors is True and vector.mag < 50
    return {hashable_vector(i): aggregated_vectors[i] for i in filter(filter_func, aggregated_vectors)}
def hashable_vector(vector):
    return PatchVector(src_patch_hash=tuple(vector[0]), unit_vect=tuple(vector[1]), mag=vector[2], dst_patch_hash=tuple(vector[3]), are_neighbors=vector[4])
def filter_playthrough_vectors(aggregated_vectors, playthrough_vectors):
    filtered_playthrough_vectors = []
    for i, vectors in enumerate(playthrough_vectors):
        filtered_vectors = []
        for j, vector in enumerate(vectors):
            if (aggregated_vectors[hashable_vector(vector)] > 2
                    and vector.are_neighbors is True
                    and vector.mag < 50):
                filtered_vectors.append(vector)
        filtered_playthrough_vectors.append(filtered_vectors)
    return filtered_playthrough_vectors
def aggregate_playthrough_vectors(playthrough_vectors):
    start_time = time.time()
    aggregated = defaultdict(int)
    for i, vectors in enumerate(playthrough_vectors):
        for j, vector in enumerate(vectors):
            aggregated[hashable_vector(vector)] += 1
    return aggregated
def get_playthrough_neighbors_index(playthrough_pix_to_patch_index: np.array, ntype: bool):
    playthrough_neighbor_index = _get_playthrough_neighbors_index(playthrough_pix_to_patch_index, ntype)
    playthrough = []
    for i, neighbor_index in enumerate(playthrough_neighbor_index):
        temp = defaultdict(dict)
        for x, y in neighbor_index:
            temp[x][y] = True
        playthrough.append(temp)
    return playthrough
@jit(nopython=True)
def _get_playthrough_neighbors_index(playthrough_pix_to_patch_index: np.array, ntype: bool):
    playthrough_neighbors = []
    for pix_to_patch_index in playthrough_pix_to_patch_index:
        my_neighbors = [(0, 0)]
        my_neighbors.pop()
        for x in range(len(pix_to_patch_index)):
            for y in range(len(pix_to_patch_index[x])):
                current_pix = pix_to_patch_index[x][y]
                neighbors = neighboring_points(x, y, pix_to_patch_index, ntype)
                for neighbor in neighbors:
                    nx = neighbor[0]
                    ny = neighbor[1]
                    neighbor_pix = pix_to_patch_index[nx][ny]
                    if current_pix != neighbor_pix:
                        my_neighbors.append((current_pix, neighbor_pix))
        my_neighbors = list(set(my_neighbors))
        playthrough_neighbors.append(my_neighbors)
    return playthrough_neighbors
#@jit(nopython=True)
def are_patches_neighbors(patch1, patch2, frame_neighbor_index):
    return frame_neighbor_index[patch1.patch_index].get(patch2.patch_index, False)
def encode_playthrough_vectors(playthrough_patches, playthrough_pix_to_patch_index, playthrough_kdtrees, playthrough_neighbor_index):
    playthrough_vectors = []
    for i, kdtree in enumerate(playthrough_kdtrees):
        start_time = time.time()
        frame_vectors = []
        frame_patches = playthrough_patches[i]
        frame_neighbor_index = playthrough_neighbor_index[i]
        pix_to_patch_index = playthrough_pix_to_patch_index[i]
        for j, src_patch in enumerate(frame_patches):
            patch_coords = Point(src_patch.first_x, src_patch.first_y)
            unit_vectors, magnitudes, dst_patch_index = normalize_knn(kdtree, [patch_coords], k=len(frame_patches))
            coords = [kdtree.data[k] for k in dst_patch_index[0]]

            dst_patches = [get_patch_from_pix_coord(int(coord[0]), int(coord[1]), frame_patches, pix_to_patch_index) for coord in coords]

            for k, dst_patch in enumerate(dst_patches):
                frame_vectors.append(
                    PatchVector(
                            src_patch[0],
                            unit_vectors[0][k],
                            magnitudes[0][k],
                            dst_patch[0],
                            are_patches_neighbors(src_patch, dst_patch, frame_neighbor_index)
                    )
                )
        #print('4b', time.time() - start_time)
        playthrough_vectors.append(frame_vectors)
    return playthrough_vectors
def get_patch_from_pix_coord(x, y, frame_patches, pix_to_patch_index):
    frame_index = pix_to_patch_index[x][y]
    return frame_patches[frame_index]
def populate_kdtrees(playthrough_patches):
    pool = mp.Pool(8)
    playthrough_coords = [[(patch.first_x, patch.first_y)  for patch in frame_patches] for frame_patches in playthrough_patches]
    return pool.map(cKDTree, playthrough_coords)
def encode_playthrough_patches(playthrough_features, playthrough_masks, playthrough_hashes):
    playthrough_patches = []
    for i, frame_features in enumerate(playthrough_features):
        frame_patches = []
        frame_masks = playthrough_masks[i]
        frame_hashes = playthrough_hashes[i]
        for j, patch_feature, in enumerate(frame_features):
            frame_patches.append(Patch(frame_hashes[j], frame_masks[j], j, *patch_feature))
        playthrough_patches.append(frame_patches)
    return playthrough_patches
def parse_and_hash_playthrough(game, play_through_number, ntype, img_count=None):
    pool = mp.Pool(8)
    pt = get_playthrough(play_through_number, game=game)['raw']

    parsed_frames_masks_and_hashes = [f for f in par(pt, img_count=img_count, ntype=ntype, pool=pool)]
    frame_features = [i[0] for i in parsed_frames_masks_and_hashes]
    masks = [i[1] for i in parsed_frames_masks_and_hashes]
    hashes = [i[2] for i in parsed_frames_masks_and_hashes]
    pix_to_patch_index = [i[3] for i in parsed_frames_masks_and_hashes]
    return frame_features, masks, hashes, pix_to_patch_index
@jit(nopython=True)
def patch_hash(patch: np.array):
    #print(len(patch))
    sx = patch.shape[0]
    sy = patch.shape[1]
    area = sx * sy
    patch_hash = np.zeros((math.ceil(float(area + 16) / 64.0)), dtype=np.uint64)

    sx_hash = np.zeros((8), dtype=np.uint64)
    for i in range(8):
        sx_hash[i] = 1 if 0b10000000 & sx else 0
        sx = sx << 1

    sy_hash = np.zeros((8), dtype=np.uint64)
    for i in range(8):
        sy_hash[i] = 1 if 0b10000000 & sy else 0
        sy = sy << 1

    p64 = np.concatenate((patch.astype(np.uint64).flatten(), sx_hash, sy_hash))
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
def quick_parse(img: np.array, ntype: bool, shrink_mask: bool = True, run_assertions: bool = False):
    if img.shape[-1] == 4:
        visited = (img[:, :, 3] == 0).astype(np.ubyte)
    else:
        visited = np.zeros(img.shape[:-1], dtype=np.ubyte)
    pix_to_patch_index = np.zeros(img.shape[:2], dtype=np.int32)
    patches_as_features = [(0, 0, 0, 0, 0, 0, 0)]  # Give numba's type inference something to work with
    patches_as_features.pop()
    masks = [np.zeros((1, 1), dtype=np.ubyte)]  # numba hint
    masks.pop()
    hashes = []
    for x in range(len(img)):
        for y in range(len(img[x])):
            if visited[x][y] == True:
                continue
            seed = (x, y)
            patch = [(x, y)]  # Hint for numba's type inference system
            stack = [patch.pop()]
            visited[x][y] = True
            pix_to_patch_index[x][y] = len(patches_as_features)
            while len(stack) > 0:
                current_pixel = stack.pop()
                patch.append(current_pixel)
                nbr_coords = neighboring_points(current_pixel[0], current_pixel[1], img, ntype)
                for i, j in nbr_coords:
                    if visited[i][j] == False and np.array_equal(img[i][j], img[x][y]):
                        pix_to_patch_index[i][j] = len(patches_as_features)
                        stack.append((i, j))
                        visited[i][j] = True
            patch_arr = np.array(patch, dtype=np.ubyte)
            x_arr = patch_arr[:, 0]
            y_arr = patch_arr[:, 1]

            if run_assertions is True:
                assert(x_arr.shape[0] == patch_arr.shape[0])
                assert(y_arr.shape[0] == patch_arr.shape[0])
                for i in range(len(x_arr)):
                    assert(patch_arr[i][0] == x_arr[i])
                    assert(patch_arr[i][1] == y_arr[i])

            x1, y1 = min(x_arr), min(y_arr),
            w, h = (max(x_arr) + 1) - x1 , (max(y_arr) + 1) - y1
            mask = np.zeros(img.shape[:-1], dtype=np.ubyte)
            for i in range(len(patch_arr)):
                nx = patch_arr[i][0] - x1
                ny = patch_arr[i][1] - y1
                if shrink_mask is True:
                    mask[nx][ny] = 1
                else:
                    mask[patch_arr[i][0]][patch_arr[i][1]] = 1
            if shrink_mask is True:
                mask = mask[:w, :h].copy()
            else:
                mask = mask.copy()
            hashes.append(patch_hash(mask))
            masks.append(mask)
            patches_as_features.append((seed[0], seed[1], len(patch), x1, y1, w, h))

    return np.array(patches_as_features, dtype=np.uint32), masks, hashes, pix_to_patch_index
def par(pt, img_count, ntype, start=0, pool=None):
    p_func = partial(quick_parse, ntype=ntype, shrink_mask=True, run_assertions=False)
    if img_count is None:
        img_count = len(pt)
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
def normalize_knn(kdt, features, k=None):
    if k is None:
        k = len(features)
    if len(kdt.data) == 1:
        return np.zeros((1, k)), np.zeros((1, k, 2))
    dist_to_dest_patch, dest_patch_index = kdt.query(features, k=k)
    mags = dist_to_dest_patch[:, 1:]
    all_coords = np.array([kdt.data[dest_patch_index[i]] for i in range(dest_patch_index.shape[0])])
    src_patch_coords = all_coords[:, 0]
    dst_patch_coords = all_coords[:, 1:]
    for i, coord in enumerate(dst_patch_coords):
        dst_patch_coords[i] = coord - src_patch_coords[i]
    flat_coords = dst_patch_coords.reshape((dst_patch_coords.shape[0] * dst_patch_coords.shape[1], 2))
    unit_vects = preprocessing.normalize(flat_coords).reshape(dst_patch_coords.shape)
    dest_patch_index = dest_patch_index[:, 1:]
    return unit_vects, mags, dest_patch_index
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
    suvs, smags = [i[0] for i in snorm], [i[1] for i in snorm]
    sfeatures = []
    for i in range(len(sprites)):
        sfeatures.append(populate_features(sprite_full_features[i], smags[i], suvs[i]))

    mario_sprite, mario_sprite_point_cloud, mario_kdt = sprites[1], sfeatures[1], sprite_kdt[1]
    muv, mmag, _ = normalize_knn(sprite_kdt[1], [i[:2] for i in mario_sprite_point_cloud], k=8)

    pt = get_playthrough(play_number, game=game)['raw']
    img_count = len(pt)
    print('img_count', img_count)
    sprites = [cv2.imread(f'./sprites/sprites/SuperMarioBros-Nes/{i}.png', cv2.IMREAD_UNCHANGED) for i in [0, 3, 35]]
    # ensure quick_parse function is compiled  is
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
    tuv, tmag, _ = norm[0]
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


    mario_sprite, mario_cloud, mario_kdt = sprites[1], sprite_clouds[1], sprite_kdt[1]
    muv, mmag, _ = normalize_knn(sprite_kdt[1], mario_cloud[:, :2], k=8)

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
    tuv, tmag, _ = normalize_knn(tree, features, k=12)
    total_time += time.time() - start
    print('normalization', time.time() - start, 'average', float(time.time() - start) / float(img_count))

    # Match points
    start = time.time()
    matches = []
    smatches = np.zeros((len(mario_cloud)), dtype=np.bool8)
    for i, tf in enumerate(full_features):
        #print([match_features(full_features[i], tmag[i], tuv[i], f, m, u) for f, m, u in zip(mario_cloud, mmag, muv)])
        matches.append([match_feature(full_features[i], tmag[i], tuv[i], f, m, u, p=True) for f, m, u in zip(mario_cloud, mmag, muv)])
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

    ti6 = np.array([
        [r, r, r, r, r, w, g, g, g, g, g],
        [r, r, r, r, r, w, g, g, g, g, g],
        [r, r, r, r, r, w, g, g, g, g, g],
        [r, r, r, r, r, w, g, g, g, g, g],
        [r, r, r, r, r, w, g, g, g, g, g],
        [b, b, b, b, b, w, m, m, m, m, m],
        [b, b, b, b, b, w, m, m, m, m, m],
        [b, b, b, b, b, w, m, m, m, m, m],
        [b, b, b, b, b, w, m, m, m, m, m],
        [b, b, b, b, b, w, m, m, m, m, m],
    ])
    tg6b = FrameGraph(ti6, ds=ds)
    for k, node in enumerate(tg6b.patches):
        old_p_hash = bin(hash(node))
        raw_patch = node.patch._patch._patch.astype(np.int64)
        phash = patch_hash(raw_patch)
        new_p_hash = bin(phash[0])
        assert(old_p_hash == new_p_hash)


    new_patch_list, masks, hashes, pix_to_patch_index = quick_parse(ti6, False, True)

    #def encode_playthrough_patches(playthrough_features, playthrough_masks, playthrough_hashes):

    encoded_patches = encode_playthrough_patches([new_patch_list], [masks], [hashes])
    expected_pix_to_patch_index = np.array([
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2],
        [3, 3, 3, 3, 3, 1, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 1, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 1, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 1, 4, 4, 4, 4, 4],
        [3, 3, 3, 3, 3, 1, 4, 4, 4, 4, 4],
    ])
    assert(np.array_equiv(pix_to_patch_index, expected_pix_to_patch_index))
    playthrough_neighbor_index = get_playthrough_neighbors_index([pix_to_patch_index], True)
    expected_neighbors = [
        [0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1],
        [0, 1, 0, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
    ]
    for j, patches in enumerate(product(playthrough_neighbor_index[0], repeat=2)):
        i1, i2 = patches[0], patches[1]
        patch1, patch2 = encoded_patches[0][i1], encoded_patches[0][i2],
        assert(expected_neighbors[i1][i2] == are_patches_neighbors(patch1, patch2, playthrough_neighbor_index[0]))

# features: [(first_pixel_x, first_pixel_y, area, bounding_box_x1, bounding_box_y1, w, h)]
if __name__ == '__main__':
    #test_new_hashing_function()
    algo_v1()
