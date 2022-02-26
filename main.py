from extract import extract
import tensorflow as tf
import numpy as np

result = extract(fr"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\image.png", [245, 220, 154])
result = np.reshape(result, (1, 28, 28, 1))
model = tf.keras.models.load_model(r'letter_classifier/letter_model')
prediction = model.predict(result)
print(type(prediction))
print(prediction)
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
index = np.argmax(prediction)
print(alphabet[index])