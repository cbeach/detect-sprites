import argparse
import gzip
from multiprocessing import Pool
import pickle
import sys
import time

from data_store import PatchDB
from sprite_util import get_image_list, get_playthrough, load_indexed_playthrough

from frame import Frame
from patch_graph import PatchGraph

parser = argparse.ArgumentParser()
parser.add_argument('-p, --play', type=int, required=True, help="the play number that you want to index", dest='play_number')
args = parser.parse_args()
start = time.time()
play_len = 0

def index_frame(t):
    i, img, write = t

    f = Frame(img)
    pg = PatchGraph(f)
    now = time.time()
    avg = (now - start) / float(i + 1)

    if write is True:
        start_write = time.time()
        with gzip.GzipFile(f'db/SuperMarioBros-Nes/{args.play_number}/{i}.pickle', 'wb') as fp:
            pickle.dump(pg, fp)
        print(i)
    #    print(f'{i}: {now - start}: {avg}: w {time.time() - start_write}: {avg * (len(play_len) - i)}')
    #else:
    #    print(f'{i}: {now - start}: {avg}: w-0: {avg * (len(play_len) - i)}')


def index_playthrough(play_number, sample_size=None, write=True):
    play_through = get_playthrough(play_number=play_number)
    pl = list(zip(range(len(play_through)), play_through, [True] * len(play_through)))
    global play_len
    play_len = len(play_through)
    #print(pl)
    with Pool(8) as p:
        p.map(index_frame, pl)


if __name__ == '__main__':
    db = PatchDB('db/')
    index_playthrough(args.play_number, write=True)
    db.write()
