from functools import partial, lru_cache
import hashlib
import multiprocessing as mp
import os
import pickle
from random import choice, choices

from .quick_parse import quick_parse

from tinydb import TinyDB, Query

DATA_DIR = f'{os.environ["HOME"]}/dev/data/game_playing'
DATASTORE_DIR = os.path.join(DATA_DIR, 'data_store')

initial_state = {
    1: {
        'dependency_hashes': {
            'simple_patch/step_1.py': None,
            'simple_patch/utils.py': None,
            'simple_patch/quick_parse.py': None,
            'simple_patch/patch_hash.py': None,
        },
        'updated': True,
        'step': 1,
    }
}

def proc_cache(step_number, db):
    def dec(func):
        Step = Query()
        state = db.search(Step.step == step_number)
        if len(state) == 0:
            init = initial_state[step_number]
            for k, v in init['dependency_hashes'].items():
                md5 = hashlib.md5()
                with open(k, 'rb') as fp:
                    md5.update(fp.read())
                    init['dependency_hashes'][k] = md5.hexdigest()
            db.insert(init)
        else:
            state = state[0]

        def proc_func(*args, **kwargs):
            if state['updated'] is True:
                return func(*args, **kwargs)
            else:
                with open(os.path.join(DATASTORE_DIR, f'{step_number}.pickle'), 'rb') as fp:
                    return pickle.load(fp)
        return proc_func
    return dec

def ser(pt, img_count, ntype, start=0):
    patches = []
    masks = []
    p_func = partial(quick_parse, ntype=ntype, shrink_mask=True, run_assertions=False)
    for i, img in enumerate(pt[start:start+img_count]):
        p, m = p_func(img)
        patches.append(p)
        masks.append(m)
    return patches, masks
def par(pt, img_count, ntype, start=0, pool=None):
    p_func = partial(quick_parse, ntype=ntype, shrink_mask=True, run_assertions=False)
    if img_count is None:
        img_count = len(pt)
    if pool is None:
        pool = mp.Pool(8)
    return pool.map(p_func, pt[start:start+img_count])



def patch_hash_to_int(hsh):
    b = hsh.tobytes()
    i = int.from_bytes(b, byteorder='little')
    return i

def vector_space_hash(src_patch_hash, dst_patch_hashes, unit_vects, mags):
    hashes_as_ints = list(map(patch_hash_to_int, dst_patch_hashes))
    hsh = (*src_patch_hash.flatten().tolist(),)
    tmp = list(sorted(zip(unit_vects, mags, hashes_as_ints), key=lambda x: x[2]))
    for i, t in enumerate(tmp):
        hsh += (*t[0], t[1], t[2])
    return hsh

