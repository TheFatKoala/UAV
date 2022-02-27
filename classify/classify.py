import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# https://www.kaggle.com/smeschke/four-shapes
letter_df = pd.read_csv(r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\letter_classifier\letter_data.csv",
                 names=[str(i) for i in range(785)])

letter_df['0'] = 0  # assign label of letter_df to 0

shape_df = pd.read_csv(r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\shape_data\shape_data.csv",
                       names=[str(i) for i in range(785)])

shape_df['0'] = 1  # assign label of shape_df to 1

frames = [letter_df, shape_df]
df = pd.concat(frames)

x_train, x_test, y_train, y_test = train_test_split(df.drop(columns="0"), df["0"], test_size=0.16,
                                                    random_state=19, stratify=df["0"])

enc = OneHotEncoder(handle_unknown='ignore')
y_train = pd.DataFrame(enc.fit_transform(y_train.values.reshape(-1, 1)).toarray())
y_test = pd.DataFrame(enc.fit_transform(y_test.values.reshape(-1, 1)).toarray())

x_train = x_train.to_numpy()
x_test = x_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

x_train = np.reshape(x_train, (-1, 28, 28)) / 255
x_test = np.reshape(x_test, (-1, 28, 28)) / 255

x_train[x_train > 0] = 1.0  # changes grayscale to binary (black and white)
x_test[x_test > 0] = 1.0  # changes grayscale to binary

y_train = np.reshape(y_train, (-1, 2))  # one hot encoding, [1, 0] is letter, [0, 1] is shape
y_test = np.reshape(y_test, (-1, 2))

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
outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, min_delta=0.05)  # early stopping -> monitors val loss

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classify_model")
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=64, epochs=76, validation_data=(x_test, y_test), callbacks=[callback])
model.save("classify_model")
