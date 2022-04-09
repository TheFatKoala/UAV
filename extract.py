import cv2
import numpy as np
import tensorflow as tf


def extract(image_path, letter_color, shape_color) -> np.array:  # color = [R, G, B]
    # returns (letter image, shape image) as tuple, as black and white (0, 255)
    letter_image = None
    shape_image = None
    lower = np.array(letter_color, dtype="uint8") - 10
    upper = np.array(letter_color, dtype="uint8") + 10
    image = cv2.imread(fr'{image_path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(image, lower, upper)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    result = cv2.bitwise_and(gray, gray, mask=mask)
    thresh, result = cv2.threshold(result, 90, 255, cv2.THRESH_BINARY)
    letter_image = cv2.resize(result, (28, 28))



    #shape_image = cv2.GaussianBlur(shape_image, (5, 5), cv2.BORDER_DEFAULT)
    #shape_image = cv2.blur(shape_image, (2, 2))
    lower = np.array(shape_color, dtype="uint8") - 15
    upper = np.array(shape_color, dtype="uint8") + 15
    shape_image = image
    mask = cv2.inRange(shape_image, lower, upper)
    gray = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    shape_image = cv2.bitwise_and(gray, gray, mask=mask)
    shape_image = cv2.blur(shape_image, (25, 25))
    thresh, shape_image = cv2.threshold(shape_image, 90, 255, cv2.THRESH_BINARY)
    shape_image = cv2.resize(shape_image, (98, 98))


    # result = cv2.resize(result, (200, 200))


    #TODO: Center image
    shape_image = cv2.resize(shape_image, (98, 98))
    #return result  # np.array
    cv2.imshow("Image", shape_image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return letter_image, shape_image

def classify_extract(image_path, color):
    lower = np.array(color, dtype="uint8") - 10
    upper = np.array(color, dtype="uint8") + 10
    image = cv2.imread(fr'{image_path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(image, lower, upper)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    result = cv2.bitwise_and(gray, gray, mask=mask)
    thresh, result = cv2.threshold(result, 90, 255, cv2.THRESH_BINARY)
    result = cv2.resize(result, (28, 28))
    return result
#extract([245, 220, 154])  # letter
#extract([206, 197, 205])  # shape