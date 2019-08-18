import cv2
import numpy as np

ss = cv2.imread('./sprite_sheet.png')
sprite_zero_zero = (1, 80)


def main(ss):
    big_mario = []
    lil_mario = []
    for i in range(21):
        x_1, x_2 = (1  , 33)
        y_1, y_2 = (80 + (i * 17), 96 + (i * 17))
        big_mario.append(ss[x_1:x_2, y_1:y_2])

        x_1, x_2 = (33  , 49)
        y_1, y_2 = (80 + (i * 17), 96 + (i * 17))

        if i < 14:
            lil_mario.append(ss[x_1:x_2, y_1:y_2])

    for i, img in enumerate(big_mario):
        cv2.imwrite('./big_mario_{}.png'.format(i), img)
    for i, img in enumerate(lil_mario):
        cv2.imwrite('./lil_mario_{}.png'.format(i), img)

print(ss.shape)
main(ss)
