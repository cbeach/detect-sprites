from glob import glob
import gzip
import itertools
import pickle
import random
import sys
import time

import cv2
import jsonpickle
import numpy as np

from patch_graph import FrameGraph
from sprite_util import show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors
import db
from db.data_store import DataStore

def byte_to_list(b):
    """
        Convert a number in range(0, 256) (byte) to a list of bool as int [01] corresponding to the binary
        expression of the number.
        Eg:
            b =  78 = 0b01001110 = [False, True, False, False, True, True, True, False]
            b = 170 = 0b10101010 = [True, False, True, False, True, False, True, False]
    """
    if b > 255:
        raise ValueError('byte must be in range(0, 256)')
    return np.array([bool(ord(i) - 48) for i in '{:0>8b}'.format(b)], dtype=bool)

bytes_as_list = [byte_to_list(i) for i in range(256)]

def process_bind_param(values):
    """
        values is an nparray
    """
    f, t, s, fi = 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff
    x, y = values.shape
    ba = bytearray([(x & f) >> 24, (x & t) >> 16, (x & s) >> 8, (x & fi), (y & f) >> 24, (y & t) >> 16, (y & s) >> 8, (y & fi)])

    counter = b = 0
    for i in values.flatten():
        b = b << 1
        b += i

        if counter == 7:
            ba.append(b)
            counter = 0
            b = 0
            continue
        else:
            counter += 1

    if counter is not 0:
        b = b << (8 - counter)
        ba.append(b)

    return ba

def process_result_value(value):
    x = y = 0
    x |= value.pop(0) << 24
    x |= value.pop(0) << 16
    x |= value.pop(0) << 8
    x |= value.pop(0)

    y |= value.pop(0) << 24
    y |= value.pop(0) << 16
    y |= value.pop(0) << 8
    y |= value.pop(0)

    length = x * y
    if length < 8:
        arr = bytes_as_list[value[0]][:length]
    else:
        arr = np.zeros((length), dtype=bool)
        for i, b in enumerate(value[:-1]):
            ind = i * 8
            arr[ind:ind + 8] = bytes_as_list[b]

        mod = length % 8
        if mod == 0:
            arr[-8:] = bytes_as_list[value[-1]]
        else:
            arr[- mod:] = bytes_as_list[value[-1]][:mod]


    arr = arr.reshape((x, y))
    return arr

def test_serialization(expected, test_number):
    res = process_bind_param(expected)
    ba = bytearray(res)
    actual = process_result_value(res)
    intermediate = ''
    if np.array_equal(expected, actual) is False:
        print(test_number, 'bad')
        print('actual:\n', actual)
        print(len(ba))
        #for i, b in enumerate(ba[8:]):
        #    print(i, b)
        #    intermediate += '{:0>8b}'.format(i)
        print('intermediate: ', intermediate)
        print('expected:\n', expected)
        print()
    else:
        print(test_number, 'good')

if __name__ == '__main__':
    a1 = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)
    a2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool)
    a3 = np.array([[0, 1]], dtype=bool)
    a4 = np.array([[0, 1, 0]], dtype=bool)
    a5 = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=bool)
    test_serialization(a1, 1)
    test_serialization(a2, 2)
    test_serialization(a3, 3)
    test_serialization(a4, 4)
    test_serialization(a5, 5)
    sys.exit(0)

    ds = DataStore()
    ds.initialize_db('./games.json')

    sys.exit(0)
    dpg = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 98, indirect=False)
    sgs = dpg.subgraphs()
    esc_key = 27
    for i, sg in enumerate(sgs):
        resp, sprite = sg.ask_if_sprite(bg_color=(255, 0, 0))
        #resp = sg.show()
        if resp == 27:
            break
