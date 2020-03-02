from collections import defaultdict, namedtuple
import numpy as np
import random
import sys

from sprites.sprite_util import show_image, show_images
from sprites.find import load_sprites
from sprites.patch_graph import FrameGraph
from sprites.db.data_store import DataStore
from sprites.patch import Node

SpriteMatch = namedtuple('SpriteMatch', ['match', 'confidence', 'obscured'])


class SpriteSet:
    def __init__(self, game: str, sprite_dir: str, ds: DataStore):
        self.ds = ds
        self.sprites = load_sprites(sprite_dir, game, ds)
        imgs = []
        for i, sprite in enumerate(self.sprites[True]):
            img = sprite.raw_frame.copy()
            for j, patch in enumerate(sprite.patches):
                if patch.frame_edge:
                    img = patch.fill_patch(img)
            imgs.append(img)
        show_images(imgs, scale=16)
        self._patch_index = self._get_patch_index()
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

        return {
            True:  {k: tuple(v) for k, v in iupsi.items()},
            False:  {k: tuple(v) for k, v in dupsi.items()},
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

    def _match_by_dfs(self, graphs, graph_anchor, sprite_index):
        pass

    def get_all_hashes(self):
        return {
            True: [hash(n) for n in self._patch_index[True].keys()],
            False: [hash(n) for n in self._patch_index[False].keys()],
        }





