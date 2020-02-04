import numpy as np
import cv2

from .patch import Node, Patch

class Sprite:
    def __init__(self, graphlet=None, bg_color=None, palette=None, path=None):
        if graphlet is None and bg_color is None and palette is None:
            self.load(path)
        else:
            self._graphlet_init(graphlet, bg_color, palette)

    def _graphlet_init(self, graphlet, bg_color, palette):
        sorted_nodes = sorted([node for node in graphlet.nodes], key=lambda n: hash(n.patch))
        self.patches = [i.patch for i in sorted_nodes]
        self.hashes = [hash(p) for p in self.patches]
        self.sprite = graphlet.palettized.copy().astype('int32')
        ox, oy = graphlet.bb[0][0], graphlet.bb[0][1]

        for i, node in enumerate(sorted_nodes):
            for x, y in node.coord_list():
                self.sprite[x - ox][y - oy] = i

        self.adjacency_matrix = am = np.zeros([len(self.hashes)] * 2, dtype=bool)
        for i, row in enumerate(self.adjacency_matrix):
            left = sorted_nodes[i]
            for j, cell in enumerate(row):
                right = sorted_nodes[j]
                am[i][j] = any([nbr is right for nbr in left.get_neighbors()])

    def to_image(self, palette):
        palette = np.array([list(c) + [255] for c in palette], dtype='uint8')
        palette[0][3] = 0
        sx, sy = self.sprite.shape
        img = np.zeros((sx, sy, 4), dtype='uint8')
        sprite = self.sprite.copy() + 1
        for x, row in enumerate(img):
            for y, pix in enumerate(row):
                img[x][y] = palette[sprite[x][y]]
        return img

    def __hash__(self):
        return int(f'{self.sprite.shape[0]}{self.sprite.shape[1]}{"".join([str(i) for i in self.sprite.flatten() + 1])}')

    def save(self, path):
        np.savez(path, hashes=self.hashes, sprite=self.sprite, adjacency_matrix=self.adjacency_matrix)

    def load(self, path):
        data = np.load(path)
        self.hashes = list(data['hashes'])
        self.patches = [Patch._PATCHES[i] for i in self.hashes]
        self.sprite = data['sprite']
        self.adjacency_matrix = data['adjacency_matrix']

    def __eq__(self, other):
        return np.array_equal(self.hashes, other.hashes) and np.array_equal(self.adjacency_matrix, other.adjacency_matrix) and np.array_equal(self.sprite, other.sprite)
