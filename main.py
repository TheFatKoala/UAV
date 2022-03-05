from extract import extract
import tensorflow as tf
import numpy as np
import cv2

# [206, 197, 205]

def classify(img_path, color):
    img = extract(fr"{img_path}", color)
    img = np.reshape(img, (1, 28, 28, 1))
    classify_model = tf.keras.models.load_model(r'classify/classify_model')
    prediction = classify_model.predict(img)
    classify_labels = ["Letter", "Shape"]
    print(f"Classify Model Prediction: Object is {classify_labels[np.argmax(prediction)]}")

def letter_classify(img):
    img = np.reshape(img, (1, 28, 28, 1))
    classify_model = tf.keras.models.load_model(r'letter_classifier/letter_model')
    prediction = classify_model.predict(img)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    print(f"Letter Model Prediction: Letter is {alphabet[np.argmax(prediction):np.argmax(prediction)+1]}")

def shape_classify(img):
    img = np.reshape(img, (1, 98, 98, 1))
    img = np.repeat(img, 3, axis=3)
    classify_model = tf.keras.models.load_model(r'shape_classifier/shape_model')
    prediction = classify_model.predict(img)
    shape_labels = ["circle", "semicircle", "quartercircle", "triangle",
          "square", "rectangle", "trapezoid", "pentagon", "hexagon",
          "heptagon", "octagon", "star", "cross"]
    print(prediction)
    print(f"Shape Model Prediction: Shape is {shape_labels[np.argmax(prediction)]}")

def main():
    img_path = r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\image.png"
    letter_color = [245, 220, 154]
    shape_color = [206, 197, 205]
    #classify(img_path, letter_color)  # should return letter
    classify(img_path, shape_color)  # should return shape
    letter_classify(img_path, letter_color)  # should return N
    shape_classify(img_path, shape_color)  # should return circle

def test():
    img_path = r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\image.png"
    letter_color = [245, 220, 154]
    shape_color = [206, 197, 205]
    letter, shape = extract(img_path, letter_color, shape_color)
    #img_path = r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\shape_data\shapes\circle.png" #
    #shape = cv2.imread(fr'{img_path}') #
    #shape = cv2.resize(shape, (98, 98)) #
    #shape = cv2.cvtColor(shape, cv2.COLOR_BGR2RGB) #
    #shape = cv2.cvtColor(shape, cv2.COLOR_BGR2GRAY) #
    print(shape.shape)
    letter_classify(letter)
    shape_classify(shape)


test()

