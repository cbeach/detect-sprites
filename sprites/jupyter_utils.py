import cv2
import numpy as np

from IPython.display import Image
from IPython.display import display as ipydisplay

def cv2_to_display(img):
    _, png_image = cv2.imencode('.png', img)
    return Image(data=png_image.tostring())

def display_image(img, scale=1.0):
    ipydisplay(cv2_to_display(cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)))

def display_images(img, scale=1.0):
    for i in img:
        ipydisplay(cv2_to_display(cv2.resize(i, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)))
