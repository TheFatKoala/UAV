import tensorflow as tf
import pandas as pd
from scipy import ndimage
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv(r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\letter_classifier\letter_data.csv",
                 names=[str(i) for i in range(785)])
del df['0']
df = df.to_numpy()
df = np.reshape(df, (-1, 28, 28))
print(df.shape)
print(np.size(df, axis=0))
labels = np.zeros((np.size(df, axis=0), 1))
for i in range(np.size(df, axis=0)):
    rotation = random.random() * 360
    df[i] = ndimage.rotate(df[i], rotation, reshape=False)
    labels[i] = rotation

df = np.reshape(df, (-1, 784))
#labels = np.reshape(df, (-1, 1))

print(df.shape)
print(labels.shape)

df = pd.DataFrame(df)
labels = pd.DataFrame(labels)

x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.16,
                                                    random_state=19)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = np.reshape(x_train, (-1, 28, 28)) / 255
x_test = np.reshape(x_test, (-1, 28, 28)) / 255

x_train[x_train > 0] = 1.0  # changes grayscale to binary (black and white)
x_test[x_test > 0] = 1.0  # changes grayscale to binary

y_train = np.reshape(y_train, (-1, 1)) / 360
y_test = np.reshape(y_test, (-1, 1)) / 360

x = np.concatenate((x_train, x_test))
y = np.concatenate((y_train, y_test))

model = tf.keras.models.load_model("letter_rotation")

output = model.predict(x)
correct = 0
total = 0
for i in range(42000):
    print(i)
    if output[i][0] * 360 - 5 < y[i][0] * 360 < output[i][0] * 360 + 5:
        correct += 1
    total += 1

print(f"Accuracy: {(correct/total)*100}%")