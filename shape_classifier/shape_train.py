import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format/code

df = pd.read_csv(r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\shape_data\shape_data.csv",
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

x_train[x_train > 0] = 1.0  # changes grayscale to binary (black and white)
x_test[x_test > 0] = 1.0  # changes grayscale to binary

y_train = np.reshape(y_train, (-1, 12))  # 12 different shapes, one hot encoded
y_test = np.reshape(y_test, (-1, 12))

inputs = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(3, 3, activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(3, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(3, 3, activation="relu")(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(363, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
x = tf.keras.layers.BatchNormalization()(x)
outputs = tf.keras.layers.Dense(12, activation="softmax")(x)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, min_delta=0.05)  # early stopping -> monitors val loss
callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\shape_classifier\shape_model",
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
model = tf.keras.Model(inputs=inputs, outputs=outputs, name="shape_model")
#model = tf.keras.models.load_model("shape_model")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=64, epochs=200, validation_data=(x_test, y_test), callbacks=[callback])
#model.save("shape_model")






