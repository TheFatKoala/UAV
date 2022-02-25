import tensorflow as tf
import pandas as pd
from scipy import ndimage
import numpy as np
import random

model = tf.keras.models.load_model('model')

df = pd.read_csv("train.csv")
del df['label']
df = df.to_numpy()
df = np.reshape(df, (42000, 28, 28))
labels = np.zeros((42000, 1))
for i in range(42000):
    print(i)
    rotation = random.random() * 360
    df[i] = ndimage.rotate(df[i], rotation, reshape=False)
    labels[i] = rotation

test_x = df / 256
test_y = labels / 360
#with tf.device("cpu:0"):
output = model.predict(test_x)
correct = 0
total = 0
for i in range(42000):
    print(i)
    if output[i][0] - 1 < test_y[i][0] < output[i][0] + 1:
        correct += 1
    total += 1

print(f"Accuracy: {(correct/total)*100}%")

