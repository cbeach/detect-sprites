import numpy as np
import cv2

def neighboring_points(x, y, arr, indirect=True):
    max_x, max_y = arr.shape[:2]
    neighbors = []
    if x > 0 and y > 0 and indirect is True:
        neighbors.append((x-1, y-1))
    if y > 0:
        neighbors.append((x, y-1))
    if x < max_x - 1 and y > 0 and indirect is True:
        neighbors.append((x+1, y-1))

    if x > 0:
        neighbors.append((x-1, y))
    #if x > 0 and np.array_equal(frame[x][y], frame[x][y]):
    #    neighbors.append((x, y))
    if x < max_x - 1:
        neighbors.append((x+1, y))

    if x > 0 and y < max_y - 1 and indirect is True:
        neighbors.append((x-1, y+1))
    if y < max_y - 1:
        neighbors.append((x, y+1))
    if x < max_x - 1 and y < max_y - 1 and indirect is True:
        neighbors.append((x+1, y+1))

    return neighbors

def show_image(img):
    cv2.imshow('frame', img)
    cv2.waitKey(0)
