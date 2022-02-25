import tensorflow as tf
import pandas as pd
from scipy import ndimage
import numpy as np
import random
import pickle


#model = tf.keras.applications.EfficientNetB0()
df = pd.read_csv("train.csv")
del df['label']
df = df.to_numpy()
df = np.reshape(df, (42000, 28, 28))
labels = np.zeros((42000, 1))
for i in range(42000):
    rotation = random.random() * 360
    df[i] = ndimage.rotate(df[i], rotation, reshape=False)
    labels[i] = rotation

train_x = df[500:] / 256
train_y = labels[500:] / 360

test_x = df[:500] / 256
test_y = labels[:500] / 360

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
outputs = tf.keras.layers.Dense(1, activation="tanh")(x)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
model.fit(x=train_x, y=train_y, batch_size=64, epochs=100, validation_data=(test_x, test_y))
model.save("model")






