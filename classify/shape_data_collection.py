import pandas as pd
from PIL import Image
import numpy as np
import os
import time
from os import listdir

# circle = 0
# square = 1
# star = 2
# triangle = 3

# turn png into csv
folder_dir = r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\k_means\shapes"
matrix_list = []
for i, shape in enumerate(["circle", "square", "star", "triangle"]):
    for image in os.listdir(folder_dir + fr"\{shape}"):
        if image.endswith(".png"):  # check if the image ends with png
            img = Image.open(fr"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\k_means\shapes\{shape}\{image}")
            img = img.resize((28, 28))
            matrix = np.array(img, dtype=np.uint8)
            matrix[matrix == 255] = 5
            matrix[matrix == 0] = 255
            matrix[matrix == 5] = 0
            matrix = matrix.flatten()
            matrix = np.insert(matrix, 0, i, axis=0)  # inserts label 0 for circle
            matrix = matrix.tolist()
            matrix_list.append(matrix)

matrix_list = np.array(matrix_list)
matrix_list = np.asarray(matrix_list)
print(matrix_list.shape)
np.savetxt("shape_data.csv", matrix_list, delimiter=",", fmt='%i')

