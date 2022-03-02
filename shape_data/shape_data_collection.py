import cv2
import pandas as pd
from PIL import Image
import numpy as np
import os
from scipy import ndimage
from PIL import Image, ImageOps
import random
import time
from os import listdir

# circle = 0
# square = 1
# star = 2
# triangle = 3

# turn png into csv
folder_dir = r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\shape_data\shapes"
matrix_list = []
labels = ["circle", "semicircle", "quartercircle", "triangle",
          "square", "rectangle", "trapezoid", "pentagon", "hexagon",
          "heptagon", "octagon", "star", "cross"]

for i, image in enumerate(os.listdir(folder_dir)):
    img = Image.open(fr"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\shape_data\shapes\{image}")
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    curr_image = img.copy()
    for _ in range(10000):  # amount of samples
        rotated_image = ndimage.rotate(curr_image, random.random()*360, reshape=True)
        src_points = np.float32([[0, 0], [0, 28],
                                 [28, 0], [28, 28]])
        warped_points = np.float32([[0+random.random()*2, 0+random.random()*2], [0+random.random()*2, 28-random.random()*2],
                                   [28-random.random()*2, 0+random.random()*2], [28-random.random()*2, 28-random.random()*2]])
        warp_transform = cv2.getPerspectiveTransform(src_points, warped_points)
        warped_img = cv2.warpPerspective(rotated_image, warp_transform, (28, 28))
        warped_img = cv2.resize(warped_img, (28, 28))
        #cv2.imshow('image', warped_img)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        if image.endswith(".png"):  # check if the image ends with png
            matrix = np.array(warped_img, dtype=np.uint8)
            matrix[matrix > 0] = 255
            matrix = matrix.flatten()
            matrix = np.insert(matrix, 0, i, axis=0)  # inserts label (0 for circle), (1 for semicircle, etc)
            matrix = matrix.tolist()
            matrix_list.append(matrix)


matrix_list = np.array(matrix_list)
matrix_list = np.asarray(matrix_list)
print(matrix_list.shape)
np.savetxt("shape_data2.csv", matrix_list, delimiter=",", fmt='%i')

