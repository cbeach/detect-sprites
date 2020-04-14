from collections import namedtuple
import unittest
import pytest as pt
from ..sprite_util import neighboring_points as nbrs
from ..sprite_util import flattened_neighboring_points as f_nbrs

import numpy as np

class TestNeighboringPoints(unittest.TestCase):
    def test_neighboring_points(self):
        unit = namedtuple('unit', ['args', 'expected'])
        test_arr = np.array([
            [[ 0], [ 1], [ 2], [ 3], [ 4], [ 5], [ 6], [ 7], [ 8], [ 9]],
            [[10], [11], [12], [13], [14], [15], [16], [17], [18], [19]],
            [[20], [21], [22], [23], [24], [25], [26], [27], [28], [29]],
            [[30], [31], [32], [33], [34], [35], [36], [37], [38], [39]],
            [[40], [41], [42], [43], [44], [45], [46], [47], [48], [49]],
            [[50], [51], [52], [53], [54], [55], [56], [57], [58], [59]],
            [[60], [61], [62], [63], [64], [65], [66], [67], [68], [69]],
            [[70], [71], [72], [73], [74], [75], [76], [77], [78], [79]],
            [[80], [81], [82], [83], [84], [85], [86], [87], [88], [89]],
            [[90], [91], [92], [93], [94], [95], [96], [97], [98], [99]]], dtype=np.uint8)

        units = [
            unit([0, 0, test_arr, True],  [1, 10, 11]),
            unit([0, 0, test_arr, False], [1, 10]),
            unit([0, 9, test_arr, True],  [8, 18, 19]),
            unit([0, 9, test_arr, False], [8, 19]),
            unit([9, 0, test_arr, True],  [80, 81, 91]),
            unit([9, 0, test_arr, False], [80, 91]),
            unit([9, 9, test_arr, True],  [88, 89, 98]),
            unit([9, 9, test_arr, False], [89, 98]),

            unit([0, 5, test_arr, True],  [4, 6, 14, 15, 16]),
            unit([0, 5, test_arr, False], [4, 6, 15]),
            unit([5, 0, test_arr, True],  [40, 41, 51, 60, 61]),
            unit([5, 0, test_arr, False], [40, 51, 60]),

            unit([5, 9, test_arr, True],  [48, 49, 58, 68, 69]),
            unit([5, 9, test_arr, False], [49, 58, 69]),

            unit([9, 5, test_arr, True],  [84, 85, 86, 94, 96]),
            unit([9, 5, test_arr, False], [85, 94, 96]),

            unit([5, 5, test_arr, True],  [44, 45, 46, 54, 56, 64, 65, 66]),
            unit([5, 5, test_arr, False], [45, 54, 56, 65]),
        ]
        for u in units:
            self.assertEqual(sorted([test_arr[x][y] for x, y in nbrs(*u.args)]), u.expected, msg=u.args)
            self.assertEqual(sorted([test_arr[x][y] for x, y in nbrs(*u.args)]), sorted([test_arr.flatten()[i] for i in f_nbrs(*u.args)]), msg=u.args)
