from extract import extract, classify_extract
import tensorflow as tf
import numpy as np

# [206, 197, 205]


def classify(img):
    img = np.reshape(img, (1, 28, 28, 1))
    classify_model = tf.keras.models.load_model(r'classify/classify_model')
    prediction = classify_model.predict(img)
    classify_labels = ["Letter", "Shape"]
    print(f"Classify Model Prediction: Object is {classify_labels[np.argmax(prediction)]}")
    return classify_labels[np.argmax(prediction)]


def letter_classify(img):
    img = np.reshape(img, (1, 28, 28, 1))
    print(img.shape)
    classify_model = tf.keras.models.load_model(r'letter_classifier/letter_model')
    print(classify_model)
    prediction = classify_model.predict(img)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter = alphabet[np.argmax(prediction):np.argmax(prediction)+1]
    print(f"Letter Model Prediction: Letter is {letter}")
    return letter


def shape_classify(img):
    img = np.reshape(img, (1, 98, 98, 1))
    img = np.repeat(img, 3, axis=3)
    classify_model = tf.keras.models.load_model(r'shape_classifier/shape_model')
    classify_model.summary()
    prediction = classify_model.predict(img)
    shape_labels = ["circle", "semicircle", "quartercircle", "triangle",
          "square", "rectangle", "trapezoid", "pentagon", "hexagon",
          "heptagon", "octagon", "star", "cross"]
    print(prediction)
    shape = shape_labels[np.argmax(prediction)]
    print(f"Shape Model Prediction: Shape is {shape}")
    return shape


def main(img_path, color1, color2):
    #img_path = r"C:\Users\hi2kh\OneDrive\Documents\GitHub\Machine-Learning\image.png"
    #letter_color = [245, 220, 154]
    #shape_color = [206, 197, 205]
    img1 = classify_extract(img_path, color1)
    if classify(img1) == "Letter":  # if image 1 is a letter
        letter_color = color1  # letter color
        shape_color = color2  # shape color
    else:
        letter_color = color2  # letter color
        shape_color = color1  # shape color

    print(letter_color)
    print(shape_color)
    letter_img, shape_img = extract(img_path, letter_color, shape_color)
    letter = letter_classify(letter_img)  # should return N
    shape = shape_classify(shape_img)  # should return circle

def test():
    #img_path = r"C:\Users\hi2kh\Documents\GitHub\Machine-Learning\image2.png"
    #letter_color = [161, 124, 215]#[245, 220, 154]
    #shape_color = [206, 197, 205]
    img_path = r"C:\Users\hi2kh\Documents\GitHub\Machine-Learning\image2.png"
    letter_color = [161, 124, 215]
    shape_color = [72, 103, 21]
    print("E")
    letter, shape = extract(img_path, letter_color, shape_color)
    print(shape.shape)
    letter_classify(letter)
    #shape_classify(shape)


#test()
main(r"C:\Users\hi2kh\Documents\GitHub\Machine-Learning\image2.png", [72, 103, 21], [161, 124, 215])

