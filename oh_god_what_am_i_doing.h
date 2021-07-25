//Point = namedtuple('Point', ['x', 'y'])
typedef struct {
    char x;
    char y;
} Point;
typedef struct {
    int width;
    int height;
} Size;
typedef struct {
    Point tlc;
    Size size;
} BoundingBox;
//Patch = namedtuple('Patch', ['hash', 'mask', 'patch_index', 'first_x', 'first_y', 'area', 'bbx1', 'bby1', 'w', 'h'])
typedef unsigned long *Hash;
typedef bool **Mask;
typedef struct {
    Hash hash;
    Mask mask;
    char index;
    Point first_pixel;
    int area;
    BoundingBox bb;
} Patch;
//PatchVector = namedtuple('PatchVector', ['src_patch_hash', 'unit_vect', 'mag', 'dst_patch_hash', 'are_neighbors', 'src_index', 'dst_index'])
typedef struct {
    Patch *src;
    Patch *dst;
    float unit_vect;
    float mag;
    bool are_neighbors;
} PatchVector;
//PatchNeighborVectorSpace = namedtuple('PatchNeighborVectorSpace', ['src_patch_index', 'src_patch_hash', 'unit_vects', 'mags', 'dst_patch_indexes', 'dst_patch_hashes'])
typedef struct {
    Patch *src;
    Patch **dst;
    int vector_count;
    float *unit_vects;
    float *mags;
} PatchNeighborVectorSpace;
