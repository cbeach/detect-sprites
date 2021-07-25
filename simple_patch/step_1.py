import multiprocessing as mp
from sprites.sprite_util import neighboring_points, get_playthrough, show_image, show_images
import os
from random import sample, choice, shuffle
import multiprocessing as mp
from .utils import par, DATA_DIR, proc_cache
import cv2
import numpy as np

from tinydb import TinyDB, Query
from .quick_parse import quick_parse

db = TinyDB(os.path.join(DATA_DIR, 'data_store/incremental_data.db'))

#@proc_cache(1, db)
def parse_and_hash_playthrough(game, play_through_number, ntype, img_count=None, img_number=None, img_path=None, random_image=False):
    pt = get_playthrough(play_through_number, game=game)['raw']
    if img_number is not None:
        parsed_frames_masks_and_hashes = [[f for f in quick_parse(pt[img_number], ntype=ntype)]]
    elif img_path is not None:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:, :, :-1]
        parsed_frames_masks_and_hashes = [[f for f in quick_parse(img, ntype=ntype)]]
    elif random_image is True:
        pool = mp.Pool(8)
        if img_count is not None:
            shuffle(pt)
            parsed_frames_masks_and_hashes = [f for f in par(pt, img_count=img_count, ntype=ntype, pool=pool)]
        else:
            parsed_frames_masks_and_hashes = [[f for f in quick_parse(choice(pt), ntype=ntype)]]
    else:
        pool = mp.Pool(8)
        parsed_frames_masks_and_hashes = [f for f in par(pt, img_count=img_count, ntype=ntype, pool=pool)]
    frame_features = [i[0] for i in parsed_frames_masks_and_hashes]
    masks = [i[1] for i in parsed_frames_masks_and_hashes]
    hashes = [i[2] for i in parsed_frames_masks_and_hashes]
    pix_to_patch_index = [i[3] for i in parsed_frames_masks_and_hashes]
    return frame_features, masks, hashes, pix_to_patch_index
