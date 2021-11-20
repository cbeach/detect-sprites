import math
from dataclasses import dataclass, field

import numpy as np
from nptyping import NDArray, UInt8, Int64, Float
from typing import NamedTuple, List, Any

from .utils import vector_space_hash

Hash = NDArray[Int64]
Mask = NDArray[UInt8]


class Patch(NamedTuple):
    hsh: Hash
    mask: Mask
    index: int
    first_x: int
    first_y: int
    area: int
    bbx1: int
    bby1: int
    w: int
    h: int


class PatchVector(NamedTuple):
    src_patch_hash: Hash
    unit_vect: float
    mag: float
    dst_patch_hash: Hash
    are_neighbors: bool
    src_index: int
    dst_index: int


@dataclass(unsafe_hash=True)
class Knns:
    indexes: NDArray[UInt8]
    mags: NDArray[Float]
    unit_vects: NDArray[Float]


@dataclass(unsafe_hash=True)
class PatchNeighborVectorSpace:
    src_index: int
    src_hash: Hash
    frame_number: int
    unit_vects: List[float] = field(default_factory=list)
    mags: List[float] = field(default_factory=list)
    dst_indexes: List[int] = field(default_factory=list)
    dst_hashes: List[Hash] = field(default_factory=list)
    vector_space_hash: List[int] = field(default_factory=list)
    all_neighbors: bool = field(default_factory=lambda: True)

    def patch_list(self, frame_patches, sorted_order=False):
        if sorted_order == False:
            return [frame_patches[i] for i in self.dst_indexes]
        else:
            return [frame_patches[i] for i in sorted(self.dst_indexes)]

    def to_image(self, frame_patches, frame):
        bb = self.bounding_box(frame_patches)
        width = bb.bottom_right_corner.x - bb.top_left_corner.x
        height = bb.bottom_right_corner.y - bb.top_left_corner.y
        img = np.zeros((width, height, 3), dtype=np.uint8)
        patches = self.patch_list(frame_patches, sorted_order=True)
        for i, patch in enumerate(patches):
            for x, row in enumerate(patch.mask):
                color = frame[patch.first_x][patch.first_y]
                for y, pix in enumerate(row):
                    offset = Point(
                        patch.bbx1 - bb.top_left_corner.x + x,
                        patch.bby1 - bb.top_left_corner.y + y
                    )
                    if pix == 1:
                        img[offset.x][offset.y] = color
        return img

    def bounding_box(self, frame_patches):
        patch_list = self.patch_list(frame_patches)
        min_x, min_y, max_x, max_y = math.inf, math.inf, -1, -1
        for patch in patch_list:
            min_x = patch.bbx1 if patch.bbx1 < min_x else min_x
            min_y = patch.bby1 if patch.bby1 < min_y else min_y
            max_x = patch.bbx1 + patch.w if patch.bbx1 + patch.w > max_x else max_x
            max_y = patch.bby1 + patch.h if patch.bby1 + patch.h > max_y else max_y
        return BoundingBox(Point(min_x, min_y), Point(max_x, max_y))

    def append(self, new_dest_patch, unit_vect, mag):
        self.dst_indexes.append(new_dest_patch.index)
        self.dst_hashes.append(new_dest_patch.hsh)
        self.unit_vects.append(unit_vect)
        self.mags.append(mag)
        self.vector_space_hash = vector_space_hash(
            self.src_hash,
            self.dst_hashes,
            self.unit_vects,
            self.mags)


class Point(NamedTuple):
    x: int
    y: int


@dataclass(unsafe_hash=True)
class BoundingBox:
    top_left_corner: Point
    bottom_right_corner: Point


@dataclass(unsafe_hash=True)
class Environment:
    playthrough: Any = None
    playthrough_features: Any = None
    playthrough_masks: Any = None
    playthrough_hashes: Any = None
    playthrough_pix_to_patch_index: Any = None
    playthrough_patches: Any = None
    playthrough_kdtrees: Any = None
    playthrough_neighbor_index: Any = None
    playthrough_vector_space: Any = None
    aggregated_vector_space: Any = None
    playthrough_knns: Any = None
    vector_space_expansions: Any = None
