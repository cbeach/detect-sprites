import math
from numba import jit
import numpy as np

@jit(nopython=True)
def patch_hash(patch: np.array):
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
    mini_hash_length = 1
    for i, pix in enumerate(p64.flatten()):
        mini_hash = mini_hash << 1
        mini_hash_length += 1
        if pix:
            mini_hash += 1

        if first_iteration is True and i == 62:  # 62 is due to the leading 1
            patch_hash[patch_hash_index] = mini_hash
            mini_hash = 0
            patch_hash_index = i // 64
            first_iteration = False
        elif patch_hash_index != i // 64: # This will trigger on the first bit after patch_hash_index rolls over
            patch_hash_index = i // 64
            patch_hash[patch_hash_index] = mini_hash
            mini_hash_length = mini_hash = 0

    if mini_hash_length > 0:
        patch_hash[patch_hash_index] = mini_hash

    return patch_hash
