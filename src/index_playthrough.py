import argparse
import gzip
from multiprocessing import Pool, Value
import pickle
import sys
import time

from data_store import PatchDB
from sprite_util import get_image_list, get_playthrough, load_indexed_playthrough, get_db_path, ensure_dir

from patch_graph import FrameGraph

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--play', type=int, required=True, help="the play number that you want to index", dest='play_number')
parser.add_argument('-ro', '--read-only', required=False, default=False, action='store_true', help='Do not write the indexed play to disk', dest='read_only')
parser.add_argument('-c', '--cores', required=False, type=int, default=8, help='level of parallelism. use zero for sequential evaluation', dest='cores')
args = parser.parse_args()
counter = Value('I', 0)
length = 0

def index_frame(t):
    i, img, read_only = t

    ipg, dpg = FrameGraph(img), FrameGraph(img, indirect=False)

    if read_only is False:
        start_write = time.time()
        with gzip.GzipFile(f'{get_db_path(args.play_number)}/{i}.pickle', 'wb') as fp:
            pickle.dump({'indirect': ipg, 'direct': dpg}, fp)

    global counter, length
    counter.value += 1
    print(str(100 * (counter.value / length)) + '%')

def index_playthrough(play_number, sample_size=None, read_only=False, cores=8):
    play_through = get_playthrough(play_number=play_number)
    sample_size =  sample_size if sample_size is not None else len(play_through)
    global length
    length = sample_size
    pl = list(zip(range(sample_size), play_through[:sample_size], [read_only] * sample_size))

    ensure_dir(f'{get_db_path(args.play_number)}/')
    if cores:
        with Pool(cores) as p:
            p.map(index_frame, pl)
    else:
        for i in pl:
            index_frame(i)

if __name__ == '__main__':
    db = PatchDB('db/')
    index_playthrough(args.play_number, read_only=args.read_only, cores=args.cores)
    db.write()
