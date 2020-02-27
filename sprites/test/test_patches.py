import os
import sys

import cv2
import numpy as np
import pytest as pt

from ..patch import Patch, frame_edge_nodes, frame_edge_node, background_node, background_nodes, Node
from ..patch_graph import FrameGraph
from ..sprite_util import show_images, show_image, get_image_list, get_playthrough, load_indexed_playthrough, sort_colors
from ..db.data_store import DataStore
from ..db.models import NodeM, PatchM

TEST_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

ds = DataStore(f'{PROJECT_ROOT}/sprites/db/sqlite.db', games_path=f'{PROJECT_ROOT}/sprites/games.json', echo=False)
Patch.init_patch_db(ds=ds)

ig = cv2.imread(f'{TEST_PATH}/images/ground.png')
ir = cv2.imread(f'{TEST_PATH}/images/repeating_ground.png')

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

    grnd_img = cv2.imread('./ground.png')
    ifg_grnd = FrameGraph(grnd_img, indirect=False, bg_color=[248, 148, 88], ds=ds)
    rep_grnd_img = cv2.imread('./repeating_ground.png')
    rep_ifg_grnd = FrameGraph(rep_grnd_img, indirect=False, ds=ds)
    degen_rep_grnd_img = cv2.imread('./degenerate_repeating_ground.png')
    degen_rep_ifg_grnd = FrameGraph(degen_rep_grnd_img, indirect=False, ds=ds)

def test_node_equallity():
    Node.set_comparison_context()
    for i, j in zip(sorted(fgg.patches, key=lambda a: a.ch()), sorted(fgg1.patches, key=lambda a: a.ch())):
        assert(i == j)

    fgg1.patches.reverse()
    for i, j in zip(fgg.patches, fgg1.patches) :
        assert(i != j)

    g__hash = [(hash(i), i) for i in fgg.patches if not i.is_special()]
    r__hash = [(hash(i), i) for i in fgr.patches if not i.is_special()]

    Node.set_comparison_context(color=False, offset=False)
    for g, r in zip(g__hash, r__hash):
        if g[0] == r[0]:
            print(g[0], r[0])
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

#def test_isomorphism():
#    grnd_img = cv2.imread('./ground.png')
#    ifg_grnd = FrameGraph(grnd_img, indirect=False, bg_color=[248, 148, 88], ds=ds)
#    rep_grnd_img = cv2.imread('./repeating_ground.png')
#    rep_ifg_grnd = FrameGraph(rep_grnd_img, indirect=False, ds=ds)
#    degen_rep_grnd_img = cv2.imread('./degenerate_repeating_ground.png')
#    degen_rep_ifg_grnd = FrameGraph(degen_rep_grnd_img, indirect=False, ds=ds)
#    cont = True
#    for ic, i in enumerate(rep_ifg_grnd.patches):
#        if ic == 3:
#            print(i.neighbors)
#            img = i.fill_neighborhood(rep_ifg_grnd.raw_frame)
#            resp = show_image(img, scale=8.0)
#            if resp == 27:
#                sys.exit(0)
#        continue
#
#        for jc, j in enumerate(ifg_grnd.patches):
#            #print(ic, jc)
#            #print(i.neighbors, j.neighbors)
#            if i == j:
#                try:
#                    print(ic, i.neighborhood_similarity(j))
#                    img = i.fill_neighborhood(rep_ifg_grnd.raw_frame)
#                    resp = show_image(img, scale=8.0)
#                    if resp == 27:
#                        sys.exit(0)
#                    #cont = False
#                    break
#                except AssertionError as e:
#                    print('exception')
#                    print(i, bin(i.master_hash(color=True)), i.color, [k.master_hash(color=True) for k in i.neighbors])
#                    print(j, bin(j.master_hash(color=True)), j.color, [k.master_hash(color=True) for k in j.neighbors])
#                    a = i.fill_patch(rep_grnd_img)
#                    for k in i.neighbors:
#                        a = k.fill_patch(a, color=(255, 0, 0))
#                    b = j.fill_patch(grnd_img)
#                    for k in j.neighbors:
#                        b = k.fill_patch(b, color=(255, 0, 0))
#                    show_images((a, b), scale=12.0)
#                    raise(e)
#        if cont is False:
#            break

def test_sprite_to_image():
    sg = Sprite(path=f'./sprites/sprites/ground0.npz')
    assert(np.array_equal(ig, sg.to_image(img=ig, bg_color=fgt.bg_color)[:, :, :-1]))
