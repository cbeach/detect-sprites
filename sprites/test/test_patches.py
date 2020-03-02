from glob import glob
import os

import cv2
import numpy as np

from ..patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes, Node
from ..patch_graph import FrameGraph
from ..sprite_util import show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors, ensure_dir
from ..db.data_store import DataStore
from ..db.models import NodeM, PatchM

TEST_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

ds = DataStore(f'{PROJECT_ROOT}/sprites/db/sqlite.db', games_path=f'{PROJECT_ROOT}/sprites/games.json', echo=False)
Patch.init_patch_db(ds=ds)

ig = cv2.imread(f'{TEST_PATH}/test_cases/SuperMarioBros-Nes/images/ground.png')
ir = cv2.imread(f'{TEST_PATH}/test_cases/SuperMarioBros-Nes/images/repeating_ground.png')

fgg = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
fgg1 = FrameGraph(ig, bg_color=[248, 148, 88], ds=ds)
fgr = FrameGraph(ir, ds=ds)

dfgg = FrameGraph(ig, indirect=False, bg_color=[248, 148, 88], ds=ds)
dfgg1 = FrameGraph(ig, indirect=False, ds=ds)
dfgr = FrameGraph(ir, indirect=False, ds=ds)
fgt = FrameGraph.from_raw_frame('SuperMarioBros-Nes', 1000, 124, indirect=False, ds=ds)

def test_patch_parsing():
    r = np.array([0, 0, 255], dtype=np.uint8)
    g = np.array([0, 255, 0], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    c = np.array([255, 255, 0], dtype=np.uint8)
    m = np.array([255, 0, 255], dtype=np.uint8)

    w = np.array([255, 255, 255], dtype=np.uint8)
    a = np.array([128, 128, 128], dtype=np.uint8)

    ti1 = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
    ])

    tg1 = FrameGraph(ti1, bg_color=np.array([0,0,0]), ds=ds)
    assert(len(tg1.patches) == 10)

    ti2 = np.array([
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
        [r, g, b, c, m, r, g, b, c, m],
    ])
    tg2 = FrameGraph(ti2, bg_color=np.array([0,0,0]), ds=ds)
    assert(len(tg2.patches) == 10)

    ti3 = np.array([
        [r, r, r, r, r, r, r, r, r, r],
        [g, g, g, g, g, g, g, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, w, w, w, w, c, c, c],
        [m, m, m, w, a, a, w, m, m, m],
        [r, r, r, w, a, a, w, r, r, r],
        [g, g, g, w, w, w, w, g, g, g],
        [b, b, b, b, b, b, b, b, b, b],
        [c, c, c, c, c, c, c, c, c, c],
        [m, m, m, m, m, m, m, m, m, m],
    ])
    tg3 = FrameGraph(ti3, bg_color=np.array([0,0,0]), ds=ds)
    assert(len(tg3.patches) == 16)

    ti4 = np.array([
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [r, r, r, r, r, w, r, r, r, r, r],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
        [b, b, b, b, b, w, b, b, b, b, b],
    ])
    tg4 = FrameGraph(ti4, bg_color=w, ds=ds)
    sg4 = tg4.subgraphs()
    assert(len(tg4.patches) == 4)
    assert(len(sg4) == 2)

    ti5 = np.array([
        [r, r, w, r, r],
        [r, r, w, r, r],
        [b, b, w, b, b],
        [b, b, w, b, b],
    ])
    tg5 = FrameGraph(ti5, bg_color=w, ds=ds)
    sg5 = tg5.subgraphs()
    assert(len(tg5.patches) == 4)
    assert(len(sg5) == 2)

def test_node_comparison():
    Node.set_comparison_context()
    for i, j in zip(sorted(fgg.patches, key=lambda a: a.ch()), sorted(fgg1.patches, key=lambda a: a.ch())):
        assert(i == j)

    fgg1.patches.reverse()
    for i, j in zip(fgg.patches, fgg1.patches) :
        assert(i != j)

    g__hash = [(hash(i), i) for i in fgg.patches if not i.is_special()]
    r__hash = [(hash(i), i) for i in fgr.patches if not i.is_special()]

    Node.set_comparison_context(color=False, offset=False)
    for i, h in enumerate(zip(g__hash, r__hash)):
        g, r = h
        if g[0] == r[0]:
            assert(g[1] == r[1])

    g_ohash = [(i.master_hash(offset=True, color=False), i) for i in fgg.patches if not i.is_special()]
    r_ohash = [(i.master_hash(offset=True, color=False), i) for i in fgr.patches if not i.is_special()]

    Node.set_comparison_context(color=False, offset=True)
    for g, r in zip(g_ohash, r_ohash):
        if g[0] == r[0]:
            assert(g[1] == r[1])

    gc_hash = [(i.master_hash(offset=False, color=True), i) for i in fgg.patches if not i.is_special()]
    rc_hash = [(i.master_hash(offset=False, color=True), i) for i in fgr.patches if not i.is_special()]

    Node.set_comparison_context(color=True, offset=False)
    for g, r in zip(gc_hash, rc_hash):
        if g[0] == r[0]:
            assert(g[1] == r[1])

    gcohash = [(i.master_hash(offset=True, color=True), i) for i in fgg.patches if not i.is_special()]
    rcohash = [(i.master_hash(offset=True, color=True), i) for i in fgr.patches if not i.is_special()]

    Node.set_comparison_context(color=True, offset=True)
    for g, r in zip(gcohash, rcohash):
        if g[0] == r[0]:
            assert(g[1] == r[1])

def test_relative_offset_hash():
    for i in sorted(fgg.patches):
        for j in sorted(fgg1.patches):
            for k1, k2 in zip(sorted([p for p in i.get_neighbors() if not p.is_special()]), sorted([p for p in j.get_neighbors() if not p.is_special()])):
                if not i.is_special() and not j.is_special() and i == j and k1 == k2:
                    assert(i.master_hash(offset=True, color=True, relative=k1) == j.master_hash(offset=True, color=True, relative=k2))

def test_relative_offset():
    target_chash = 9222229577035039944
    target_nbr_ohash = 2146476057
    img = dfgg.raw_frame

    target_node = dfgg.patches[[i.ch() for i in dfgg.patches].index(target_chash)]
    nbr = target_node.get_neighbors()[[i.oh() for i in target_node.get_neighbors()].index(target_nbr_ohash)]
    assert(target_node.get_relative_offset(nbr) == (-1, 8))

def test_images():
    def color_frame_edges(sprite: FrameGraph):
        img = sprite.raw_frame.copy()
        for patch in sprite.patches:
            if patch.frame_edge is True:
                img = patch.fill_patch(img)
        return img

    def color_bg_edges(sprite: FrameGraph):
        img = sprite.raw_frame.copy()
        for patch in sprite.patches:
            if patch.bg_edge is True:
                img = patch.fill_patch(img)
        return img

    names = [path.split('/')[-1] for path in glob(f'{TEST_PATH}/test_cases/SuperMarioBros-Nes/images/test_images/*')]
    has_bg_color = ['repeating' in name for name in names]

    test_cases = []
    expected = []
    for i, test_name in enumerate(names):
        expected.append(cv2.imread(f'{TEST_PATH}/test_cases/SuperMarioBros-Nes/images/expected/{test_name}', cv2.IMREAD_UNCHANGED))
        test_img = cv2.imread(f'{TEST_PATH}/test_cases/SuperMarioBros-Nes/images/test_images/{test_name}', cv2.IMREAD_UNCHANGED)
        if 'background' in test_name or 'repeating' in test_name:
            if '.indirect.' in test_name:
                test_cases.append(FrameGraph(test_img.copy(), indirect=True, bg_color=test_img[0][0].copy(), ds=ds))
            elif '.direct.' in test_name:
                test_cases.append(FrameGraph(test_img.copy(), indirect=False, bg_color=test_img[0][0].copy(), ds=ds))
        else:
            if '.indirect.' in test_name:
                test_cases.append(FrameGraph(test_img.copy(), indirect=True, ds=ds))
            elif '.direct.' in test_name:
                test_cases.append(FrameGraph(test_img.copy(), indirect=False, ds=ds))

    def image_diff(source, img1, img2):
        if source.shape != img1.shape or source.shape != img2.shape:
            raise ValueError(f'all images must be the same shape - source: {source.shape}, img1: {img1.shape}, img2: {img2.shape}')

        src = source.copy()
        for x, row in enumerate(img1):
            for y, pixel in enumerate(row):
                if not np.array_equal(img1[x][y], img2[x][y]):
                    src[x][y][0] = 0
                    src[x][y][1] = 255
                    src[x][y][2] = 0
        return src

    def instrumented_assertion(i, test_name):
        try:
            assert(np.array_equal(color_func(test_cases[i])[:, :, :3], expected[i][:, :, :3]))
        except AssertionError as err:
            print('i', i, 'test_name', test_name, 'color_func', color_func, 'len(patches)', len(test_cases[i].patches),
                  'bg_color', test_cases[i].bg_color)
            cv2.imwrite(f'{TEST_PATH}/failed-tests/original-{test_name}', test_cases[i].raw_frame)
            cv2.imwrite(f'{TEST_PATH}/failed-tests/colored-{test_name}', color_func(test_cases[i]))
            cv2.imwrite(f'{TEST_PATH}/failed-tests/expected-{test_name}', expected[i])
            cv2.imwrite(f'{TEST_PATH}/failed-tests/diffed-{test_name}',
                        image_diff(test_cases[i].raw_frame[:, :, :3], color_func(test_cases[i])[:, :, :3],
                                   expected[i][:, :, :3]))
            raise err

    ensure_dir(f'{TEST_PATH}/failed-tests/')
    for i, test_name in enumerate(names):
        if 'background' in test_name:
            color_func = color_bg_edges
        elif 'frame_edge' in test_name:
            color_func = color_frame_edges

        instrumented_assertion(i, test_name)
