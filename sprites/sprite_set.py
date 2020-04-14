from collections import defaultdict, namedtuple
from glob import glob
import numpy as np
from queue import Queue
import random
import sys

from sprites.sprite_util import show_image, show_images
from sprites.find import pairs_equal
from sprites.patch_graph import FrameGraph
from sprites.db.data_store import DataStore
from sprites.patch import Node

SpriteMatch = namedtuple('SpriteMatch', ['match', 'confidence', 'obscured'])

def load_sprites(sprite_dir='./sprites/sprites/SuperMarioBros-Nes', game='SuperMarioBros-Nes', ds=None):
    isprites = []
    dsprites = []
    paths = []
    for i, path in enumerate(glob(f'{sprite_dir}/*')):
        extension = path.split('.')[-1]
        if extension == 'png':
            isprites.append(FrameGraph.from_path(path, game=game, indirect=True, ds=ds))
            dsprites.append(FrameGraph.from_path(path, game=game, indirect=False, ds=ds))
            paths.append(path)

    return {
        True: isprites,
        False: dsprites,
        'paths': paths,
    }

class SpriteSet:
    def __init__(self, game: str, sprite_dir: str, ds: DataStore):
        self.ds = ds
        self.sprites = load_sprites(sprite_dir, game, ds)
        self._patch_index = self._get_patch_index()
        self._single_node_sprites = {
            True:  [i for i, sprite in enumerate(self.sprites[True])  if len(sprite.patches) == 1],
            False: [i for i, sprite in enumerate(self.sprites[False]) if len(sprite.patches) == 1],
        }
        self._upi = self._unique_patch_index()

    def _get_patch_index(self):
        isg = self.sprites[True]  # indirect sprite graph
        dsg = self.sprites[False]  # direct sprite graph

        # [nodes] -> [[_patches]]
        patches = {
            True: [[n.patch._patch for n in s.patches] for s in isg],
            False: [[n.patch._patch for n in s.patches] for s in dsg],
        }

        # [[_patches]] -> {_patch: [int]}
        # For each _patch, a list of sprites that it appears in
        ipsi = defaultdict(list)  # indirect_patch_to_sprite_index
        for i, ps in enumerate(patches[True]):
            for p in ps:
                ipsi[p].append(i)

        dpsi = defaultdict(list)  # direct_patch_to_sprite_index
        for i, ps in enumerate(patches[False]):
            for p in ps:
                dpsi[p].append(i)

        # Modify the patch index from above so that each list of sprites is a set
        return {
            True:  {k: tuple(set(v)) for k, v in ipsi.items()},
            False: {k: tuple(set(v)) for k, v in dpsi.items()},
        }

    def _unique_patch_index(self):
        # Modify the patch index from above to include only patches that are associated with one sprite.
        iupsi = defaultdict(set)
        for k, v in self._patch_index[True].items():
            if len(v) == 1:
                iupsi[v[0]] = iupsi[v[0]].union({k})

        dupsi = defaultdict(set)
        for k, v in self._patch_index[True].items():
            if len(v) == 1:
                dupsi[v[0]] = dupsi[v[0]].union({k})

        ifiltered = defaultdict(list)
        dfiltered = defaultdict(list)
        for k, v in iupsi.items():
            sprite = self.sprites[True][k]
            #ifiltered[k] = sorted([patch for patch in sprite.patches if sprite.patches.count(patch) == 1 and patch.touches_edge() is False and patch.area() > 7], key=lambda p: p.area(), reverse=True)
            ifiltered[k] = sorted([patch for patch in sprite.patches if sprite.patches.count(patch) == 1 and patch.area() > 7], key=lambda p: p.area(), reverse=True)

        for k, v in dupsi.items():
            sprite = self.sprites[False][k]
            dfiltered[k] = sorted([patch for patch in sprite.patches if sprite.patches.count(patch) == 1 and patch.area() > 7], key=lambda p: p.area(), reverse=True)

        return {
            True:  {k: v for k, v in ifiltered.items() if len(v) > 0},
            False:  {k: v for k, v in dfiltered.items() if len(v) > 0},
        }

    def random_unique_patches(self, ntype=True):
        return {k: random.choice(v) for k, v in self._upi[ntype].items()}

    def is_match(self, graph: FrameGraph, anchor: Node, sprite_index: int):
        """
        :param graph:
        :param anchor: int: The index of the patch in graph that we want to start our search from
        :param sprite_index: int
        :return: namedtuple: SpriteMatch
        """
        ntype = graph.indirect

    def _find_matching_node(self, query_node, sprite):
        possible_matching_nodes = [patch for patch in sprite.patches if hash(patch) == hash(query_node)]

        q_nbrs = sorted([n for n in query_node.get_neighbors() if not n.touches_edge() and not n.touches_bg()], key=hash)
        for i, node in enumerate(possible_matching_nodes):
            s_nbrs = sorted([n for n in node.get_neighbors()
                             if not n.touches_edge()
                             and not n.touches_bg()], key=hash)
            for sn in s_nbrs:
                pass
                #qns =

    def _match_by_bfs(self, graph, graph_root_node: Node, sprite_index):
        """
        this method most likely garbage
        """
        ntype = graph.indirect
        sprite = self.sprites[ntype][sprite_index]

        # For now I'm assuming that this is a unique node, it may not be in the future
        sprite_root_node = sprite.patches[sprite.patches.index(graph_root_node)]

        MatchingNodes = namedtuple('MatchingNodes', ['graph', 'sprite', 'depth'])
        visited_graph = defaultdict(lambda: False)
        visited_sprite = defaultdict(lambda: False)
        matches = []
        qu = Queue()
        qu.put_nowait(MatchingNodes(graph_root_node, sprite_root_node, 0))
        depth = 0
        count = 0
        #gimg = graph_root_node.fill_patch(graph.raw_frame.copy())
        #simg = sprite_root_node.fill_patch(sprite.raw_frame.copy())
        gimg = graph.raw_frame.copy()
        simg = sprite.raw_frame.copy()
        while qu.empty() is False and depth < 5:
            graph_node, sprite_node, depth = qu.get_nowait()
            sprite_nbrs = sprite_node.get_neighbors()
            graph_nbrs = graph_node.get_neighbors()
            #gimg = graph_node.fill_patch(gimg, color=[0, 255, 0])

            for snbr in sprite_nbrs:
                print(visited_sprite[snbr.oh()])
                if visited_sprite[snbr.oh()] is True:
                    continue

                for gnbr in graph_nbrs:
                    if visited_graph[gnbr.oh()] is True:
                        continue

                    match = MatchingNodes(graph=gnbr, sprite=snbr, depth=depth + 1)
                    qu.put_nowait(match)
                    show_images((gnbr.fill_patch(gimg, color=[0, 255, 0]), snbr.fill_patch(simg, color=[0, 255, 0])), scale=3)

                    if pairs_equal((sprite_node, snbr), (graph_node, gnbr)) is True and snbr.touches_edge() is False:
                        matches.append(gnbr)
                        #gimg = gnbr.fill_patch(gimg, color=[0, 255, 0])
                        #simg = snbr.fill_patch(simg, color=[0, 255, 0])
                    #else:
                        #gimg = gnbr.fill_patch(gimg, color=[0, 0, 255])
                        #simg = snbr.fill_patch(simg, color=[0, 0, 255])

                    visited_graph[gnbr.oh()] = True
                visited_sprite[snbr.oh()] = True
            count += 1

        show_images((gimg, simg), scale=3)
        return matches

    def get_all_hashes(self):
        return {
            True: [hash(n) for n in self._patch_index[True].keys()],
            False: [hash(n) for n in self._patch_index[False].keys()],
        }





