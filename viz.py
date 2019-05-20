# -*- coding: utf-8 -*-
# File: viz.py

import numpy as np

# import cv2
import matplotlib.pyplot as plt


label_colours = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 29], [219, 219, 0], [106, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [69, 129, 180], [219, 19, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 69]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 79, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle
 

def decode_labels(mask, img_shape, num_classes):
    color_table = label_colours

    color_mat = np.array(color_table, dtype=np.float32)
    mask = np.ravel(mask)
    pred = np.zeros((mask.size, 3))
    for ii, c in enumerate(color_table):
        midx = np.where(mask == ii)[0]
        pred[midx, :] = np.reshape(c, (1, 3))

    pred = np.reshape(pred, (img_shape[0], img_shape[1], 3))
    
    return pred


def draw_predictions(img, preds, num_classes):
    """
    """
    result = decode_labels(preds, preds.shape, num_classes)
    plt.imshow(result / 255.0)
    plt.show()
