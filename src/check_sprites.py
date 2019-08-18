#!/usr/bin/env python

import json
import os
import random

import cv2

play_data_path_fmt = '/home/mcsmash/dev/data/game_playing/play_data/SuperMarioBros-Nes/frames/{}-{}.png'

#with open('small_sample.json', 'r') as fp:
with open('sprite_locations.json', 'r') as fp:
    locs = json.load(fp)

random.shuffle(locs)

def get_image(play, frame):
    return cv2.imread(play_data_path_fmt.format(play, frame))

count = 0
for i, loc in enumerate(locs):
    if 'loc' in loc:
        img = get_image(loc['play'], loc['frame'])
        rec = cv2.rectangle(img, tuple(loc['loc'][0]), tuple(loc['loc'][1]), (255, 255, 0))
        cv2.imwrite('temp/{}.png'.format(i), rec)
        count += 1
        print(count)
    if count == 1000:
        break

