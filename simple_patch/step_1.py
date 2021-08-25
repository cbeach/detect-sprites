import multiprocessing as mp
from sprites.sprite_util import neighboring_points, get_playthrough, show_image, show_images
import os
from random import sample, choice, shuffle
import multiprocessing as mp
from .utils import DATA_DIR, proc_cache
import cv2
import numpy as np

from tinydb import TinyDB, Query
from .quick_parse import quick_parse

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

db = TinyDB(os.path.join(DATA_DIR, 'data_store/incremental_data.db'))


def parse_and_hash_playthrough(game, playthrough_number, ntype, img_count=None, img_number=None, img_path=None, random_image=False, start_index=None, parrallel=True):
    if parrallel is True:
        from .utils import par as func
        pool = mp.Pool(8)
    else:
        from .utils import ser as func
    if game is not None and playthrough_number is not None:
        pt = get_playthrough(playthrough_number, game=game)['raw']
        if img_number is not None:
            parsed_frames_masks_and_hashes = [[f for f in quick_parse(pt[img_number], ntype=ntype)]]
        elif random_image is True:
            if img_count is not None:
                shuffle(pt)
                parsed_frames_masks_and_hashes = [f for f in func(pt, img_count=img_count, ntype=ntype, pool=pool)]
            else:
                parsed_frames_masks_and_hashes = [[f for f in quick_parse(choice(pt), ntype=ntype)]]
        elif start_index is not None:
            parsed_frames_masks_and_hashes = [f for f in func(pt[start_index:img_count], img_count=img_count, ntype=ntype, pool=pool)]
        else:
            parsed_frames_masks_and_hashes = [f for f in func(pt, img_count=img_count, ntype=ntype, pool=pool)]
    else:
        if img_path is not None:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:, :, :-1]
            parsed_frames_masks_and_hashes = [[f for f in quick_parse(img, ntype=ntype)]]

    frame_features = [i[0] for i in parsed_frames_masks_and_hashes]
    masks = [i[1] for i in parsed_frames_masks_and_hashes]
    hashes = [i[2] for i in parsed_frames_masks_and_hashes]
    pix_to_patch_index = [i[3] for i in parsed_frames_masks_and_hashes]
    return frame_features, masks, hashes, pix_to_patch_index
