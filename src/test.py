import numpy as np

from frame import Frame
from patch_graph import PatchGraph

def main():
    #img = cv2.imread('./test/test-examples/SuperMarioBros-Nes/test_frames/test-frame.png')
    #f = Frame(img)
    #PatchGraph(f)

    r = np.array([0, 0, 255], dtype=np.uint8)
    g = np.array([0, 255, 0], dtype=np.uint8)
    b = np.array([255, 0, 0], dtype=np.uint8)
    c = np.array([255, 255, 0], dtype=np.uint8)
    m = np.array([255, 0, 255], dtype=np.uint8)

    w = np.array([255, 255, 255], dtype=np.uint8)
    a = np.array([128, 128, 128], dtype=np.uint8)

    #ti1 = np.array([
    #    [r, r, r, r, r, r, r, r, r, r],
    #    [g, g, g, g, g, g, g, g, g, g],
    #    [b, b, b, b, b, b, b, b, b, b],
    #    [c, c, c, c, c, c, c, c, c, c],
    #    [m, m, m, m, m, m, m, m, m, m],
    #    [r, r, r, r, r, r, r, r, r, r],
    #    [g, g, g, g, g, g, g, g, g, g],
    #    [b, b, b, b, b, b, b, b, b, b],
    #    [c, c, c, c, c, c, c, c, c, c],
    #    [m, m, m, m, m, m, m, m, m, m],
    #])

    #tf1 = Frame(ti1, bg_color=np.array([0,0,0]))
    #tg1 = PatchGraph(tf1)

    #ti2 = np.array([
    #    [r, g, b, c, m, r, g, b, c, m],
    #    [r, g, b, c, m, r, g, b, c, m],
    #    [r, g, b, c, m, r, g, b, c, m],
    #    [r, g, b, c, m, r, g, b, c, m],
    #    [r, g, b, c, m, r, g, b, c, m],
    #])
    #tf2 = Frame(ti2, bg_color=np.array([0,0,0]))
    #tg2 = PatchGraph(tf2)

    #ti3 = np.array([
    #    [r, r, r, r, r, r, r, r, r, r],
    #    [g, g, g, g, g, g, g, g, g, g],
    #    [b, b, b, b, b, b, b, b, b, b],
    #    [c, c, c, w, w, w, w, c, c, c],
    #    [m, m, m, w, a, a, w, m, m, m],
    #    [r, r, r, w, a, a, w, r, r, r],
    #    [g, g, g, w, w, w, w, g, g, g],
    #    [b, b, b, b, b, b, b, b, b, b],
    #    [c, c, c, c, c, c, c, c, c, c],
    #    [m, m, m, m, m, m, m, m, m, m],
    #])
    #tf3 = Frame(ti3, bg_color=np.array([0,0,0]))
    #tg3 = PatchGraph(tf3)

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
    tf4 = Frame(ti4, bg_color=w)
    tg4 = PatchGraph(tf4)
    tg4.isolate_offset_subgraphs()

if __name__ == "__main__":
    main()
