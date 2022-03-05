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

print(df.shape)
print(labels.shape)
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.16,
                                                    random_state=19)

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = np.reshape(x_train, (-1, 28, 28, 1)) / 255
x_test = np.reshape(x_test, (-1, 28, 28, 1)) / 255
x_train = np.repeat(x_train, 3, axis=3)
x_test = np.repeat(x_test, 3, axis=3)

x_train[x_train > 0] = 1.0  # changes grayscale to binary (black and white)
x_test[x_test > 0] = 1.0  # changes grayscale to binary

y_train = np.reshape(y_train, (-1, 1)) / 360
y_test = np.reshape(y_test, (-1, 1)) / 360

#inputs = tf.keras.Input(shape=(28, 28, 3))
#x = tf.keras.layers.Normalization()(inputs)
#x = tf.keras.layers.Conv2D(3, 3, activation="relu", bias_initializer=tf.keras.initializers.Constant(0.1))(x)
#x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#x = tf.keras.layers.Normalization()(x)
#x = tf.keras.layers.Conv2D(3, 3, activation="relu", bias_initializer=tf.keras.initializers.Constant(0.1))(x)
#x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
#x = tf.keras.layers.Flatten()(x)
#x = tf.keras.layers.Normalization()(x)
#x = tf.keras.layers.Dense(363, activation="relu", bias_initializer=tf.keras.initializers.Constant(0.1))(x)
#x = tf.keras.layers.Normalization()(x)
#x = tf.keras.layers.Dense(100, activation="relu", bias_initializer=tf.keras.initializers.Constant(0.1))(x)
#x = tf.keras.layers.Normalization()(x)
#outputs = tf.keras.layers.Dense(1, activation="tanh", bias_initializer=tf.keras.initializers.Constant(0.1))(x)
inputs = tf.keras.layers.Input(shape=(28, 28, 3))
base_model = tf.keras.applications.MobileNetV2(input_tensor=inputs, weights='imagenet', include_top=False)
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1028, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='tanh')(x)

for layer in base_model.layers:
    layer.trainable = False

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, min_delta=0.01)  # early stopping -> monitors val loss
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\letter_rotation\letter_rotation",
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="letter_rotation")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=64, epochs=500, validation_data=(x_test, y_test), callbacks=[callback])
model.save("letter_rotation")

