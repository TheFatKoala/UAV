import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy import ndimage
import random

# https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/code

df = pd.read_csv(r"C:\Users\hi2kh\Documents\GitHub\Machine-Learning\letter_classifier\letter_data.csv",
                 names=[str(i) for i in range(785)])

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns="0"), df["0"], test_size=0.16,
                                                    random_state=19, stratify=df["0"])
# makes sure same freq for every label, randomizes data


enc = OneHotEncoder(handle_unknown='ignore')
y_train = pd.DataFrame(enc.fit_transform(y_train.values.reshape(-1, 1)).toarray())
y_test = pd.DataFrame(enc.fit_transform(y_test.values.reshape(-1, 1)).toarray())  # one hot encoded labels, 0 = A, 1 = B, etc.


x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = np.reshape(x_train, (-1, 28, 28)) / 255
x_test = np.reshape(x_test, (-1, 28, 28)) / 255

for i in range(np.size(x_train, axis=0)):
    rotation = random.random() * 360
    x_train[i] = ndimage.rotate(x_train[i], rotation, reshape=False)

for i in range(np.size(x_test, axis=0)):
    rotation = random.random() * 360
    x_test[i] = ndimage.rotate(x_test[i], rotation, reshape=False)

x_train[x_train > 0] = 1.0  # changes grayscale to binary (black and white)
x_test[x_test > 0] = 1.0  # changes grayscale to binary

y_train = np.reshape(y_train, (-1, 26))  # one hot encoding, [1, 0, 0, ...] = A, [0, 1, 0, ...] = B etc
y_test = np.reshape(y_test, (-1, 26))

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(3, 3, activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Normalization()(x)
x = tf.keras.layers.Conv2D(3, 3, activation="relu")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Normalization()(x)
x = tf.keras.layers.Dense(363, activation="relu")(x)
x = tf.keras.layers.Normalization()(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
x = tf.keras.layers.Normalization()(x)
outputs = tf.keras.layers.Dense(26, activation="softmax")(x)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)  # early stopping -> monitors val loss

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="letter_model")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=64, epochs=76, validation_data=(x_test, y_test), callbacks=[callback])
model.save("letter_model")






