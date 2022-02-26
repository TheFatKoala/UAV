import cv2
import numpy as np

def extract(color):  # color = [R, G, B]
    # returns processed image as black and white (0, 255)
    image = cv2.imread(r'C:\Users\hi2kh\OneDrive\Desktop\Python\Letter Recognition\image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower = np.array(color, dtype="uint8") - 10
    upper = np.array(color, dtype="uint8") + 10
    mask = cv2.inRange(image, lower, upper)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(mask, cnts, (255, 255, 255))
    result = cv2.bitwise_and(gray, gray, mask=mask)
    (thresh, result) = cv2.threshold(result, 90, 255, cv2.THRESH_BINARY)
    #TODO: Center image

    cv2.imshow("Image", result)
    cv2.waitKey()
    cv2.destroyAllWindows()

#extract([245,220,154])
extract([206,197,205])