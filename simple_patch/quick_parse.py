import numpy as np
from numba import jit
from sprites.sprite_util import neighboring_points, get_playthrough, show_image, show_images
from .patch_hash import patch_hash

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
