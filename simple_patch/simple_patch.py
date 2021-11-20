from collections.abc import Iterable
from collections import defaultdict, OrderedDict
from glob import glob
import math
import functools
from functools import partial, lru_cache
import hashlib
import psutil
import shutil

import io
from itertools import product, groupby
import multiprocessing as mp
import os
import pickle
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
from tinydb import TinyDB, Query
import progressbar
from more_itertools import chunked

from sprites.sprite_util import neighboring_points, get_playthrough, show_image, show_images
from sprites.find import fit_bounding_box
from sprites.patch_graph import FrameGraph
from simple_patch.step_1 import parse_and_hash_playthrough
from simple_patch.utils import DATA_DIR, vector_space_hash, patch_hash_to_int
from simple_patch.patch_hash import patch_hash
from simple_patch.quick_parse import quick_parse
from simple_patch.sprite_types import Patch, PatchVector, PatchNeighborVectorSpace, Knns, Environment, BoundingBox, Point
# features: [(first_pixel_x, first_pixel_y, area, bounding_box_x1, bounding_box_y1, w, h)]


from termcolor import cprint as _cprint
import inspect, io, termcolor as _tm, pprint as _pp
def formatted_file_and_lineno(depth=1): frame_stack = inspect.getouterframes(inspect.currentframe()); parrent = frame_stack[depth]; return f'{parrent.filename}:{parrent.lineno} $> '
def cpprint(obj, color='white'): buffer = io.StringIO(); _pp.pprint(obj, buffer); _tm.cprint(buffer.getvalue(), color);
def prepend_lineno(*obj, depth=1): return f'{formatted_file_and_lineno(depth + 1)}{" ".join(map(str, obj))}'
def lineno_print(*obj):
    if obj == '' or len(obj) == 0:
        _print(*obj)
    else:
        _print(prepend_lineno(*obj, depth=2))
def lineno_pprint(obj): _print(formatted_file_and_lineno(depth=2), end=''); _pp.pprint(obj)
def lineno_cprint(obj, color='white'): _tm.cprint(prepend_lineno(obj, depth=2), color)
def lineno_cpprint(obj, color='white'): _print(formatted_file_and_lineno(depth=2), end=''); _cpprint(obj, color);
_print = print; print = lineno_print; cprint = lineno_cprint; pprint = lineno_pprint; _cpprint = cpprint; cpprint = lineno_cpprint;

np.set_printoptions(threshold=10000000, linewidth=1000000000)
shrink = True

db = TinyDB(os.path.join(DATA_DIR, 'data_store/incremental_data.db'))

def get_env(*args, **kwargs):
    ntype = False
    mag_threshold = 50
    playthrough_features, playthrough_masks, playthrough_hashes, playthrough_pix_to_patch_index, playthrough \
        = parse_and_hash_playthrough(*args, **kwargs)
    playthrough_patches = encode_playthrough_patches(playthrough_features, playthrough_masks, playthrough_hashes)
    playthrough_kdtrees = populate_kdtrees(playthrough_patches)
    playthrough_neighbor_index = get_playthrough_neighbors_index(playthrough_pix_to_patch_index, ntype)
    playthrough_vector_space, playthrough_knns = encode_playthrough_vectors(
        playthrough_patches,
        playthrough_pix_to_patch_index,
        playthrough_kdtrees,
        playthrough_neighbor_index,
        mag_threshold=mag_threshold)
    aggregated_vector_space = aggregate_playthrough_vector_space(playthrough_vector_space)
    vector_space_expansions = playtime(playthrough_knns, playthrough_patches, playthrough_vector_space)
    return Environment(
        playthrough=playthrough,
        playthrough_features=playthrough_features,
        playthrough_masks=playthrough_masks,
        playthrough_hashes=playthrough_hashes,
        playthrough_pix_to_patch_index=playthrough_pix_to_patch_index,
        playthrough_patches=playthrough_patches,
        playthrough_kdtrees=playthrough_kdtrees,
        playthrough_neighbor_index=playthrough_neighbor_index,
        playthrough_vector_space=playthrough_vector_space,
        aggregated_vector_space=aggregated_vector_space,
        playthrough_knns=playthrough_knns,
        vector_space_expansions=vector_space_expansions
    )

def algo_v1(pickle_me=True):
    game = 'SuperMarioBros-Nes'
    play_number = 1000
    ntype = False
    mag_threshold = 50
    if pickle_me is True:
        print('recalculating...')

        playthrough_features, playthrough_masks, playthrough_hashes, playthrough_pix_to_patch_index, playthrough \
            = parse_and_hash_playthrough(game, play_number, ntype, img_count=10)
        save_data(playthrough_features, 'playthrough_features')
        save_data(playthrough_masks, 'playthrough_masks')
        save_data(playthrough_hashes, 'playthrough_hashes')
        save_data(playthrough_pix_to_patch_index, 'playthrough_pix_to_patch_index')
        save_data(playthrough, 'playthrough')
        playthrough_patches = encode_playthrough_patches(playthrough_features, playthrough_masks, playthrough_hashes)
        save_data(playthrough_patches, 'playthrough_patches')
        playthrough_kdtrees = populate_kdtrees(playthrough_patches)
        save_data(playthrough_kdtrees, 'playthrough_kdtrees')
        playthrough_neighbor_index = get_playthrough_neighbors_index(playthrough_pix_to_patch_index, ntype)
        save_data(playthrough_neighbor_index, 'playthrough_neighbor_index')
        playthrough_vector_space, playthrough_knns = encode_playthrough_vectors(
            playthrough_patches,
            playthrough_pix_to_patch_index,
            playthrough_kdtrees,
            playthrough_neighbor_index,
            mag_threshold=mag_threshold)
        save_data(playthrough_vector_space, 'playthrough_vector_space')
        save_data(playthrough_knns, 'playthrough_knns')
        aggregated_vector_space = aggregate_playthrough_vector_space(playthrough_vector_space)
        save_data([(key, value) for key, value in aggregated_vector_space.items()], 'aggregated_vector_space')
        vector_space_expansions = playtime(playthrough_knns, playthrough_patches, playthrough_vector_space)
        save_data(vector_space_expansions, 'vector_space_expansions')
    else:
        env = load_env()
        print('hello!')

def playtime(playthrough_knns, playthrough_patches, playthrough_vector_space):
    playthrough_frame_aggregation = []
    tmp = defaultdict(int)
    for i, fvs in enumerate(playthrough_vector_space):
        for j, vs in enumerate(fvs):
            tmp[vs.vector_space_hash] += 1
        playthrough_frame_aggregation.append(dict(tmp))

    frame_exps = []
    def visited(vs, exps):
        for k in exps:
            if (vs.src_index == k.src_index or vs.src_index in k.dst_indexes) and vs.frame_number == k.frame_number:
                return True
        return False
    for i, frame_vs in enumerate(playthrough_vector_space):
        exps = []
        frame_aggregation = playthrough_frame_aggregation[i]
        for j, vs in enumerate(frame_vs):
            if frame_aggregation[vs.vector_space_hash] > 1 or visited(vs, exps) or len(vs.dst_indexes) == 0:
                continue
            exp = PatchNeighborVectorSpace(
                frame_number=vs.frame_number,
                src_index=vs.src_index,
                src_hash=vs.src_hash.copy(),
                unit_vects=vs.unit_vects.copy(),
                mags=vs.mags.copy(),
                dst_indexes=vs.dst_indexes.copy(),
                dst_hashes=vs.dst_hashes.copy(),
                vector_space_hash=vs.vector_space_hash,
                all_neighbors=False,
            )
            temp = extend_vs(playthrough_knns, playthrough_patches, playthrough_vector_space, vs, None, exp, playthrough_frame_aggregation[i])
            exps.append(temp)
        frame_exps.append(exps)
    return frame_exps


def extend_vs(playthrough_knns, playthrough_patches, playthrough_vector_space, src_vs, next_vs_list, exp, frame_aggregation, depth=0):
    frame_number = src_vs.frame_number
    def vs_list(vs):
        temp = []
        for index in vs.dst_indexes:
            if frame_aggregation[playthrough_vector_space[frame_number][index].vector_space_hash] == 1:
                temp.append(playthrough_vector_space[frame_number][index])
        return temp

    if next_vs_list is None:
        next_vs_list = vs_list(src_vs)
    src_index = src_vs.src_index

    for next_vs in next_vs_list:
        next_index = next_vs.src_index
        next_patch = playthrough_patches[frame_number][next_index]
        if next_patch.area > 250:  # arbitrary threshold
            continue
        for j, skip_vs_src_index in enumerate(next_vs.dst_indexes):
            skip_patch = playthrough_patches[frame_number][skip_vs_src_index]
            skip_vs = playthrough_vector_space[frame_number][skip_patch.index]
            if frame_aggregation[skip_vs.vector_space_hash] == 1:
                temp = np.where(playthrough_knns[frame_number][src_index] == skip_vs_src_index)
                if skip_patch.index not in exp.dst_indexes and len(temp) == 1:
                    knn_index = temp[0]
                    skip_knn = playthrough_knns[frame_number][src_vs.src_index]
                    exp.append(
                        skip_patch,
                        skip_knn.unit_vects[knn_index],
                        skip_knn.mags[knn_index])
                    extend_vs(playthrough_knns, playthrough_patches, playthrough_vector_space, src_vs, vs_list(skip_vs), exp, frame_aggregation, depth=depth + 1)

    return exp

def save_data(data, name, base_path='./cache'):
    mypath = f'{base_path}/{name}'
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    for i, chunk in enumerate(chunked(data, 500)):
        save_path = f'{mypath}/{i}'
        print(f'saving {name} at {save_path}')
        with open(save_path, 'wb') as fp:
            pickle.dump(chunk, fp)

def load_data(name):
    data = []
    print(f'{name}/')
    for i in range(len(glob(f'{name}/*'))):
        file_path = f'{name}/{i}'
        print(f'\t{file_path}')
        with open(file_path, 'rb') as fp:
            data.extend(pickle.load(fp))
    return data

def load_env():
    data = {}
    for dataclass_field in Environment.__dataclass_fields__.keys():
        print(dataclass_field)
        data[dataclass_field] = load_data(dataclass_field)
    return Environment(**data)

def determine_bayesian_probabilities(playthrough_vector_space, aggregated_vector_space):
    grouped_vector_spaces = []

    for i in playthrough_vector_space:
         grouped_vector_spaces.append(
             {
                 j: {l.vector_space_hash: l for l in k}
                 for j, k in groupby(i, key=lambda a: patch_hash_to_int(a.src_hash))
             })
    expanded_vector_space = []
    for i, group in enumerate(grouped_vector_spaces):
        expansion = {}
        for key, value in group.items():
            if len(value) == 1:
                # iterate over the destinations
                expansion[key] = PatchNeighborVectorSpace(
                    frame_number=value[0].frame_number,
                    src_index=value[0].src_index,
                    src_hash=value[0].src_hash,
                    unit_vects=value[0].unit_vects,
                    mags=value[0].mags,
                    dst_indexes=value[0].dst_indexes,
                    dst_hashes=value[0].dst_hashes,
                    vector_space_hash=value[0].vector_space_hash,
                    all_neighbors=value[0].all_neighbors,
                )
                print(expansion[key])
                time.sleep(1)
                sys.exit(0)

    return grouped_vector_spaces
def partition_vectors(filtered_aggregated_vectors, filtered_playthrough_vectors, playthough_vector_space, aggregated_vector_space):
    """
    filtered_aggregated_vectors:
        <dict> PatchVector: int
    The number of times a vector occurs in a playthrough
    filtered_playthrough_vectors:
        <list(frame)>[<list PatchVector>]
    Every meaningful PatchVector in each frame of the playthrough.
    """
    count = 0
    for i, playthrough_vectors in enumerate(filtered_playthrough_vectors):
        """
        i: frame number
        playthrough_vectors: filtered vectors from frame i
        """
        # sort by number of occurrences in playthrough
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
def filter_aggregated_vectors(aggregated_vectors, mag_threshold=50):
    def filter_func(vector):
        return aggregated_vectors[vector] > 2 and vector.are_neighbors is True and vector.mags < mag_threshold
    return {hashable_vector(i): aggregated_vectors[i] for i in filter(filter_func, aggregated_vectors)}
def hashable_vector_space_hash(vector_space):
    return tuple(
        [
            tuple(i) if isinstance(i, Iterable) else i
            for i in vector_space.vector_space_hash
        ]
    )
def hashable_vector(vector):
    return PatchVector(src_patch_hash=tuple(vector[0]), unit_vect=tuple(vector[1]), mag=vector[2], dst_patch_hash=tuple(vector[3]), are_neighbors=vector[4], src_index=vector.src_index, dst_index=vector.dst_index)
def filter_playthrough_vectors(aggregated_vectors, playthrough_vectors):
    filtered_playthrough_vectors = []
    for i, vectors in enumerate(playthrough_vectors):
        filtered_vectors = []
        for j, vector in enumerate(vectors):
            if (aggregated_vectors[hashable_vector(vector)] > 2
                    and vector.are_neighbors is True
                    and vector.mags < 50):
                filtered_vectors.append(vector)
        filtered_playthrough_vectors.append(filtered_vectors)
    return filtered_playthrough_vectors
def compare_arrays(a1, a2):
    if np.array_equiv(a1, a2):
        return 0
    min_len = min(len(a1), len(a2))
    for i in range(min_len):
        if a1[i] < a2[i]:
            return -1
        elif a1[i] > a2[i]:
            return 1
        else:
            continue
    # At this point the shorter array is known to be a prefix of the larger array. Return comparison based on length.
    if len(a1) < len(a2):
        return -1
    elif len(a1) > len(a2):
        return 1
    else:
        raise RuntimeError('You should not be here. Arrays are equivalent, but this case was not caught during the first condition of this function.')
def aggregate_playthrough_vector_space(playthrough_vector_space):
    aggregated = defaultdict(int)
    for i, fvs in enumerate(playthrough_vector_space):
        for j, vs in enumerate(fvs):
            aggregated[vs.vector_space_hash] += 1
    return aggregated
def aggregate_playthrough_vectors(playthrough_vectors):
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
def are_patches_neighbors(patch1, patch2, frame_neighbor_index):
    return frame_neighbor_index[patch1.index].get(patch2.index, False)
def encode_playthrough_vectors(playthrough_patches, playthrough_pix_to_patch_index, playthrough_kdtrees, playthrough_neighbor_index, mag_threshold=50):
    playthrough_vector_space = []
    playthrough_knns = []
    with progressbar.ProgressBar(max_value=len(playthrough_kdtrees)) as bar:
        for i, kdtree in enumerate(playthrough_kdtrees):
            bar.update(i)
            frame_vector_space = []
            frame_knns = []
            frame_patches = playthrough_patches[i]
            frame_neighbor_index = playthrough_neighbor_index[i]
            pix_to_patch_index = playthrough_pix_to_patch_index[i]
            for j, src_patch in enumerate(frame_patches):
                if len(frame_patches) <= 1:
                    continue
                patch_coords = Point(src_patch.first_x, src_patch.first_y)
                unit_vectors, magnitudes, dst_patch_index, coords = normalize_knn(kdtree, [patch_coords], k=len(frame_patches), distance_upper_bound=50)
                if len(dst_patch_index) == 0:
                    frame_knns.append(Knns(dst_patch_index, magnitudes, unit_vectors))
                    frame_vector_space.append(PatchNeighborVectorSpace(
                        src_index=i, src_hash=src_patch[0], frame_number=i, unit_vects=unit_vectors, mags=magnitudes,
                        dst_hashes=[], dst_indexes=[], )
                    )
                    frame_vector_space[-1].vector_space_hash = vector_space_hash(src_patch.hsh,
                                                                       frame_vector_space[-1].dst_hashes,
                                                                       frame_vector_space[-1].unit_vects,
                                                                       frame_vector_space[-1].mags)
                    continue
                dst_patches = []
                for k, index in enumerate(dst_patch_index):
                    if magnitudes[k] < mag_threshold:
                        dst_patches.append(frame_patches[index])
                vector_space = PatchNeighborVectorSpace(
                    src_index=j, src_hash=src_patch[0], frame_number=i
                )
                frame_knns.append(Knns(dst_patch_index, magnitudes, unit_vectors))
                for l, dst_patch in enumerate(dst_patches):
                    neighbors = are_patches_neighbors(src_patch, dst_patch, frame_neighbor_index)
                    if neighbors is True:
                        vector_space.unit_vects.append(unit_vectors[l])
                        vector_space.mags.append(magnitudes[l])
                        vector_space.dst_hashes.append(dst_patch.hsh)
                        vector_space.dst_indexes.append(dst_patch.index)
                vector_space.vector_space_hash = vector_space_hash(src_patch.hsh, vector_space.dst_hashes, vector_space.unit_vects, vector_space.mags)
                frame_vector_space.append(vector_space)
            playthrough_vector_space.append(frame_vector_space)
            playthrough_knns.append(frame_knns)
    return playthrough_vector_space, playthrough_knns
def get_patch_from_pix_coord(x, y, frame_patches, pix_to_patch_index):
    frame_index = pix_to_patch_index[x][y]
    return frame_patches[frame_index]
def populate_kdtrees(playthrough_patches):
    pool = mp.Pool(8)
    playthrough_coords = [[(patch.first_x, patch.first_y) for patch in frame_patches] for frame_patches in playthrough_patches]

    #literal = ''
    #with open('patch_coords.h', 'w') as fp:
    #    for b, i in enumerate(playthrough_coords):
    #        print(len(i))
    #        literal += f'double coord{b}[{len(i)}][2] = ' + '{'
    #        for j in i[:-1]:
    #            literal += '{' + str(j[0]) + ', ' + str(j[1]) + '},'
    #        literal += '{' + str(i[-1][0]) + ', ' + str(i[-1][1]) + '}};\n'
    #    fp.write(literal)
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
def normalize_knn(kdt, features, k=None, distance_upper_bound=None):
    if k is None:
        k = len(features)
    if len(kdt.data) == 1:
        return np.zeros((1, k)), np.zeros((1, k, 2))
    if distance_upper_bound is None:
        dist_to_dest_patch, dest_patch_index = kdt.query(features, k=k, workers=-1)
    else:
        dist_to_dest_patch, dest_patch_index = kdt.query(features, k=k, workers=-1, distance_upper_bound=distance_upper_bound)
    length = np.count_nonzero(np.where(dest_patch_index < len(kdt.data))) + 1
    if length <= 1:
        return np.array([], dtype=np.float), np.array([], dtype=np.float), np.array([], dtype=np.uint), \
               np.array([], dtype=np.uint8)
    mags = dist_to_dest_patch[0][1:length]
    all_coords = kdt.data[dest_patch_index[0][:length], :]
    src_patch_coords = all_coords[0]
    dst_patch_coords = all_coords[1:]
    for i, coord in enumerate(dst_patch_coords):
        dst_patch_coords[i] = coord - src_patch_coords
    unit_vects = preprocessing.normalize(dst_patch_coords).reshape(dst_patch_coords.shape)
    dest_patch_index = dest_patch_index.flatten()[1:length]
    return unit_vects, mags, dest_patch_index, all_coords[1:]
def color_mask(img, mask, color=(0, 255, 0)):
    c = np.array(color, dtype=np.ubyte)
    img = img.copy()
    for x, y in zip(*np.nonzero(mask)):
        img[x][y] = c
    return img
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
    ds = DataStore(f'{PROJECT_ROOT}/../sprites/db/sqlite.db', games_path=f'{PROJECT_ROOT}/sprites/games.json', echo=False)
    r = np.array([0, 0, 255], dtype=np.uint8)
    g = np.array([0, 255, 0], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    c = np.array([255, 255, 0], dtype=np.uint8)
    m = np.array([255, 0, 255], dtype=np.uint8)

    w = np.array([255, 255, 255], dtype=np.uint8)
    k = np.array([0, 0, 0], dtype=np.uint8)
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
        old_p_hash = bin(node.patch._patch.my_hash)
        raw_patch = node.patch._patch._patch.astype(np.int64)
        phash = patch_hash(raw_patch)
        new_p_hash = bin(phash[0])
        assert(old_p_hash == new_p_hash)

    run_tests_on_frame(
        img=np.array([
            [w, w, b, b, b, w, w],
            [w, w, w, b, w, w, w],
            [w, w, w, w, w, w, w],
            [w, w, w, w, w, w, w],
            [w, w, b, w, b, w, w],
            [w, w, b, b, b, w, w],
            [w, w, b, b, b, w, w],
        ]),
        ds=ds,
        expected_pix_to_patch_index=np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 0, 2, 0, 0],
            [0, 0, 2, 2, 2, 0, 0],
            [0, 0, 2, 2, 2, 0, 0],
        ]),
        expected_neighbors=[
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ])

def compare_hashes(frame_graph):
    for k, node in enumerate(frame_graph.patches):
        raw_patch = node.patch._patch._patch.astype(np.int64)
        old_p_hash = bin(node.patch._patch.my_hash)
        phash = patch_hash(raw_patch)
        # add 16 bits for the masks shape, one byte per dimension
        # add 1 bit for the prefix '1'
        area = len(raw_patch.flatten()) + 16 + 1
        new_p_hash = '0b'
        for i in phash.flatten():
            new_p_hash += bin(int(i))[2:]

        assert(old_p_hash == new_p_hash)

def run_tests_on_frame(img, ds, expected_pix_to_patch_index=None, expected_neighbors=None):
    frame_graph = FrameGraph(img, ds=ds)
    compare_hashes(frame_graph)

    new_patch_list, masks, hashes, pix_to_patch_index = quick_parse(img, False, True)
    encoded_patches = encode_playthrough_patches([new_patch_list], [masks], [hashes])
    if expected_pix_to_patch_index is not None:
        assert(np.array_equiv(pix_to_patch_index, expected_pix_to_patch_index))

    playthrough_neighbor_index = get_playthrough_neighbors_index([pix_to_patch_index], True)
    for j, patches in enumerate(product(playthrough_neighbor_index[0], repeat=2)):
        i1, i2 = patches[0], patches[1]
        patch1, patch2 = encoded_patches[0][i1], encoded_patches[0][i2],
        if expected_neighbors is not None:
            assert(expected_neighbors[i1][i2] == are_patches_neighbors(patch1, patch2, playthrough_neighbor_index[0]))

def test_vector_space_function():
    ntype = False
    mag_threshold = 50

    playthrough_features, playthrough_masks, playthrough_hashes, playthrough_pix_to_patch_index, playthrough \
            = parse_and_hash_playthrough(None, None, ntype=ntype, img_path='sprites/sprites/SuperMarioBros-Nes/3.png')
    reference_pix_to_patch_index = np.array([
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  0,  2,  2,  2,  2,  2,  0,  0,  1,  1,  1],
        [ 0,  0,  0,  0,  0,  2,  2,  2,  2,  2,  2,  2,  2,  2,  1,  1],
        [ 0,  0,  0,  0,  0,  3,  3,  3,  4,  4,  5,  6,  7,  8,  8,  8],
        [ 0,  0,  0,  0,  9, 10,  3,  4,  4,  4,  5,  6,  6,  8,  8,  8],
        [ 0,  0,  0,  0,  9, 10,  3,  3,  4,  4,  4, 11,  6,  6,  6,  8],
        [ 0,  0,  0,  0,  9,  9,  4,  4,  4,  4, 11, 11, 11, 11, 11, 12],
        [ 0,  0,  0,  0,  0,  0,  4,  4,  4,  4,  4,  4,  4, 11, 12, 12],
        [ 0,  0, 13, 13, 13, 13, 13, 14, 15, 15, 15, 16, 17, 12, 12, 12],
        [ 0, 13, 13, 13, 13, 13, 13, 13, 18, 15, 15, 15, 18, 12, 12, 19],
        [20, 20, 13, 13, 13, 13, 13, 13, 18, 18, 18, 18, 18, 12, 12, 19],
        [20, 20, 20, 21, 18, 18, 13, 18, 18, 22, 18, 18, 23, 18, 19, 19],
        [24, 20, 25, 26, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19],
        [24, 24, 26, 26, 26, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19],
        [24, 26, 26, 26, 18, 18, 18, 18, 18, 18, 18, 27, 27, 27, 27, 27],
        [24, 26, 28, 28, 18, 18, 18, 18, 27, 27, 27, 27, 27, 27, 27, 27],
    ])
    assert(np.array_equiv(reference_pix_to_patch_index, playthrough_pix_to_patch_index[0]))
    assert(len(playthrough_features[0]) == len(playthrough_masks[0]) == len(playthrough_hashes[0]) == 29)
    playthrough_patches = encode_playthrough_patches(playthrough_features, playthrough_masks, playthrough_hashes)
    playthrough_kdtrees = populate_kdtrees(playthrough_patches)
    playthrough_neighbor_index = get_playthrough_neighbors_index(playthrough_pix_to_patch_index, ntype)
    #playthrough_vectors,
    playthrough_vector_space, _ = encode_playthrough_vectors(
        playthrough_patches,
        playthrough_pix_to_patch_index,
        playthrough_kdtrees,
        playthrough_neighbor_index,
        mag_threshold=mag_threshold)
    expected_playthrough_vector_space_dst_indexes = [
        [1, 2, 3, 4, 9, 13, 20],
        [0, 2, 8],
        [0, 1, 3, 4, 5, 6, 7, 8],
        [0, 2, 4, 10],
        [0, 2, 3, 5, 9, 11, 13, 14, 15, 16, 17],
        [2, 4, 6],
        [2, 5, 7, 8, 11],
        [2, 6, 8],
        [1, 2, 6, 7, 12],
        [0, 4, 10],
        [3, 9],
        [4, 6, 12],
        [8, 11, 17, 18, 19],
        [0, 4, 14, 18, 20, 21],
        [4, 13, 15],
        [4, 14, 16, 18],
        [4, 15, 17],
        [4, 12, 16, 18],
        [12, 13, 15, 17, 19, 21, 22, 23, 26, 27, 28],
        [12, 18, 27],
        [0, 13, 21, 24, 25],
        [13, 18, 20, 26],
        [18],
        [18],
        [20, 26],
        [20, 26],
        [18, 21, 24, 25, 28],
        [18, 19],
        [18, 26]
    ]
    actual_playthrough_vector_space_dst_indexes = [list(sorted(i.dst_indexes)) for i in playthrough_vector_space[0]]
    assert(expected_playthrough_vector_space_dst_indexes == actual_playthrough_vector_space_dst_indexes)
    temp = {
        playthrough_vector_space[0][0].vector_space_hash: True
    }
    aggregated_vector_space = aggregate_playthrough_vector_space(playthrough_vector_space)

def test_aggregation():
    ntype = False
    mario_on_small_canvas = get_env(None, None, ntype=ntype, parrallel=False, img_path='sprites/sprites/SuperMarioBros-Nes/3.png')
    expected_playthrough_vector_space_dst_indexes = [
        [1, 2, 3, 4, 9, 13, 20],
        [0, 2, 8],
        [0, 1, 3, 4, 5, 6, 7, 8],
        [0, 2, 4, 10],
        [0, 2, 3, 5, 9, 11, 13, 14, 15, 16, 17],
        [2, 4, 6],
        [2, 5, 7, 8, 11],
        [2, 6, 8],
        [1, 2, 6, 7, 12],
        [0, 4, 10],
        [3, 9],
        [4, 6, 12],
        [8, 11, 17, 18, 19],
        [0, 4, 14, 18, 20, 21],
        [4, 13, 15],
        [4, 14, 16, 18],
        [4, 15, 17],
        [4, 12, 16, 18],
        [12, 13, 15, 17, 19, 21, 22, 23, 26, 27, 28],
        [12, 18, 27],
        [0, 13, 21, 24, 25],
        [13, 18, 20, 26],
        [18],
        [18],
        [20, 26],
        [20, 26],
        [18, 21, 24, 25, 28],
        [18, 19],
        [18, 26]
    ]
    actual_playthrough_vector_space_dst_indexes = [list(sorted(i.dst_indexes)) for i in mario_on_small_canvas.playthrough_vector_space[0]]
    assert(expected_playthrough_vector_space_dst_indexes == actual_playthrough_vector_space_dst_indexes)
    assert(len(mario_on_small_canvas.playthrough_vector_space[0]) == 29)
    mosc_exps = run_playtime_tests(mario_on_small_canvas, {
        'exps_len': [1],
        'dst_indexes': [[list(range(29))]],
    })

    mario_on_large_canvas = get_env(None, None, ntype=ntype, parrallel=False, img_path='/home/mcsmash/dev/deep_thought/data_tools/preprocessing/detect-sprites/sprites/sprites/SuperMarioBros-Nes/tests/mario-on-a-large-canvas.png')
    assert(len(mario_on_large_canvas.playthrough_vector_space[0]) == 25)
    expected_playthrough_vector_space_dst_indexes = [
        [],                                     # 0
        [2, 8],                                 # 1
        [1, 3, 4, 5, 6, 7, 8],                  # 2
        [2, 4, 10],                             # 3
        [2, 3, 5, 9, 11, 12, 13, 14, 15, 16],   # 4
        [2, 4, 6],                              # 5
        [2, 5, 7, 8, 11],                       # 6
        [2, 6, 8],                              # 7
        [1, 2, 6, 7],                           # 8
        [4, 10],                                # 9
        [3, 9],                                 # 10
        [4, 6],                                 # 11
        [4, 13, 17, 19, 20],                    # 12
        [4, 12, 14],                            # 13
        [4, 13, 15, 17],                        # 14
        [4, 14, 16],                            # 15
        [4, 15, 17],                            # 16
        [12, 14, 16, 18, 20, 21, 22, 24],       # 17
        [17],                                   # 18
        [12, 20, 23],                           # 19
        [12, 17, 19, 24],                       # 20
        [17],                                   # 21
        [17],                                   # 22
        [19, 24],                               # 23
        [17, 20, 23],                           # 24
    ]
    actual_playthrough_vector_space_dst_indexes = [list(sorted(i.dst_indexes)) for i in mario_on_large_canvas.playthrough_vector_space[0]]
    assert(expected_playthrough_vector_space_dst_indexes == actual_playthrough_vector_space_dst_indexes)
    molc_exps = run_playtime_tests(mario_on_large_canvas, {
        'exps_len': [1],
        'dst_indexes': [[list(range(1, 25))]],
    })
    mario_and_goomba = get_env(None, None, ntype=ntype, parrallel=False, img_path='/home/mcsmash/dev/deep_thought/data_tools/preprocessing/detect-sprites/sprites/sprites/SuperMarioBros-Nes/tests/mario-and-a-goomba-far-apart.png')
    assert(len(mario_and_goomba.playthrough_vector_space[0]) == 35)
    expected_playthrough_vector_space_dst_indexes = [
        [],                                     # 0
        [2, 8],                                 # 1
        [1, 3, 4, 5, 6, 7, 8],                  # 2
        [2, 4, 10],                             # 3
        [2, 3, 5, 9, 11, 12, 13, 14, 15, 16],   # 4
        [2, 4, 6],                              # 5
        [2, 5, 7, 8, 11],                       # 6
        [2, 6, 8],                              # 7
        [1, 2, 6, 7],                           # 8
        [4, 10],                                # 9
        [3, 9],                                 # 10
        [4, 6],                                 # 11
        [4, 13, 17, 19, 20],                    # 12
        [4, 12, 14],                            # 13
        [4, 13, 15, 17],                        # 14
        [4, 14, 16],                            # 15
        [4, 15, 17],                            # 16
        [12, 14, 16, 18, 20, 21, 22, 24],       # 17
        [17],                                   # 18
        [12, 20, 23],                           # 19
        [12, 17, 19, 24],                       # 20
        [17],                                   # 21
        [17],                                   # 22
        [19, 24],                               # 23
        [17, 20, 23],                           # 24
        [27, 28, 30],                           # 25
        [27, 28, 29, 30, 31, 32],               # 26
        [25, 26, 29],                           # 27
        [25, 26, 31],                           # 28
        [26, 27, 30],                           # 29
        [25, 26, 29, 31],                       # 30
        [26, 28, 30],                           # 31
        [26, 33, 34],                           # 32
        [32],                                   # 33
        [32],                                   # 34
    ]
    actual_playthrough_vector_space_dst_indexes = [list(sorted(i.dst_indexes)) for i in mario_and_goomba.playthrough_vector_space[0]]

    assert(expected_playthrough_vector_space_dst_indexes == actual_playthrough_vector_space_dst_indexes)
    mandg = run_playtime_tests(mario_and_goomba, {
        'exps_len': [2],
        'dst_indexes': [[list(range(1, 25)), list(range(25, 35))]],
    })
    
def test_extension():
    sequence_path = './test/pngs/sequences/running/'
    ntype = False
    env = get_env(None, None, ntype, playthrough_path=sequence_path)
    #show_image(env.playthrough[0])
    return
    for i, exps in enumerate(env.vector_space_expansions):
        show_images([j.to_image(env.playthrough_patches[i], env.playthrough[i]) for j in exps])


def run_playtime_tests(env, expected):
    exps = playtime(env.playthrough_knns, env.playthrough_patches, env.playthrough_vector_space)
    for i, frame_exp in enumerate(exps):
        try:
            assert len(frame_exp) == expected['exps_len'][i], f'exps_len test failed.\n\ractual: {len(frame_exp)}\n\texpected: {expected["exps_len"][i]}'
        except AssertionError as e:
            for j, exp in enumerate(frame_exp):
                print(j, list(sorted(exp.dst_indexes)))
                raise e
        for j, expected_dst in enumerate(expected['dst_indexes']):
            assert len(frame_exp[j].dst_indexes) == len(expected_dst[i]), f'dst_indexes_len[{j}] test failed.\n\ractual: {len(frame_exp[j].dst_indexes)}\n\texpected: {len(expected_dst[i])}'
            assert list(sorted(frame_exp[j].dst_indexes)) == list(sorted(expected_dst[i])), f'dst_indexes_len[{j}] test failed.\n\ractual: {list(sorted(frame_exp[j].dst_indexes))}\n\texpected: {list(sorted(expected_dst[i]))}'

    return exps
