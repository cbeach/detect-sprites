
from sprite_util import neighboring_points

import numpy as np

class PatchGraph:
    def __init__(self, frame):
        self.frame = frame

        #self.hash_to_patch = {hash(i):i for i in self.frame.patches}
        self.offset_hash_to_patch = {i.hash_with_offset():i for i in self.frame.patches}

        self.hashes = sorted(list(set([hash(i) for i in self.frame.patches])))
        self.offset_hashes = sorted([i.hash_with_offset() for i in self.frame.patches])

        self.hash_index = {self.hashes[i]:i for i in range(len(self.hashes))}
        self.full_hash_index = {self.offset_hashes[i]:i for i in range(len(self.offset_hashes))}

        self.offset_hash_to_hash = {i.hash_with_offset():hash(i) for i in self.frame.patches}
        self.hash_to_offset_hash = {hash(i):i.hash_with_offset() for i in self.frame.patches}

        self.adjacency_matrix = np.zeros((len(self.hashes), len(self.hashes)), dtype=bool)
        self.full_adjacency_matrix = np.zeros((len(self.offset_hashes), len(self.offset_hashes)), dtype=bool)

        self.graph = self.build_graph()

    def build_graph(self):
        for patch in self.frame.patches:
            """
                For each patch, get the list of pixels just outside the patch's outer edge.
                Use that list of pixels to find all patches that are touching this one.
            """
            current_hash = hash(patch)
            current_offset_hash = patch.hash_with_offset()
            nbr_pixels = patch.get_neighboring_patch_pixels(self.frame)
            nbr_patches = list(set(self.frame.get_patches_by_coord(nbr_pixels)))
            for npatch in nbr_patches:
                npatch_hash = hash(npatch)
                npatch_offset_hash = npatch.hash_with_offset()
                self.adjacency_matrix[self.hash_index[current_hash]][self.hash_index[npatch_hash]] = True
                self.full_adjacency_matrix[self.full_hash_index[current_offset_hash]][self.full_hash_index[npatch_offset_hash]] = True

        #print(f'self.full_adjacency_matrix.shape: {self.full_adjacency_matrix.shape}')
        #for i, row in enumerate(self.full_adjacency_matrix):
        #    patch = self.offset_hash_to_patch[self.offset_hashes[i]]
        #    print(f'{row} {patch.bounding_box[0]} {patch.color}')
        #print()
        #print(len(self.frame.patches))
