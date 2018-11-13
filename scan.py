from libs.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import os
import time
import cv2 
import imutils
import pytesseract
from PIL import Image, ImageFont, ImageDraw

font = cv2.FONT_HERSHEY_COMPLEX
bottomLeftCornerOfText = (50, 50)
fontScale = 1
fontColor = (0, 0, 230)
lineType = 2

EDGE = 120

def nothing(x, y):
    pass 

cap = cv2.VideoCapture(0)

while(True):
    
    ret, image = cap.read()
    # image = cv2.imread('ocvi_1541504014.668524_gray.jpg')

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, EDGE)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)

        if len(approx) == 4 and cv2.contourArea(c) > 5000:
            screenCnt = approx
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
            
    cv2.imshow('Contornada', edged)
    cv2.imshow('Original', image)
    cv2.createTrackbar('Contornos', 'Contornada', 120, 200, nothing)
    
    EDGE = cv2.getTrackbarPos('trackbar', 'Contornada')
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(99) & 0xFF == ord('c'):
        if 'screenCnt' in locals():
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        else:
            warped = image
        

        text = pytesseract.image_to_string(warped, lang='por')
        print(text)
        text_image = Image.new("RGB", [500, 500])
        draw = ImageDraw.Draw(text_image)
        font = ImageFont.truetype("/Library/Fonts/Arial Unicode.ttf", 14)
        draw.text((50, 50), text, font=font)
        text_image = np.array(text_image)
        cv2.imshow('Recortada', warped)
        cv2.imshow('Original', image)
        cv2.imshow('Texto', text_image)
        
        current = str(time.time())
        os.mkdir('uploads/' + current)

        cv2.imwrite('uploads/' + current + '/recortada_' + current + '.jpg', warped)
        cv2.imwrite('uploads/' + current + '/redimensionada_' + current + '.jpg', image)
        cv2.imwrite('uploads/' + current + '/texto_' + current + '.jpg', text_image)
        cv2.imwrite('uploads/' + current + '/original_' + current + '.jpg', orig)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


cv2.destroyAllWindows()
