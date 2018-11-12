import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ret, thresh = cv2.threshold(gray, 127, 255, 1)

# contours, h = cv2.findContours(thresh, 1, 2)

while(1):
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 1)
    # contours = cv2.findContours(thresh, 1, 2)
    im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # cv2.imshow('frame', im2)


    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        
        # print(len(approx))
        if len(approx) == 4:
            print("square")
            cv2.drawContours(frame, [cnt], 0, 255, -1)
        # if len(approx) == 5:
        #     print("pentagon")
        #     cv2.drawContours(frame, [cnt], 0, 255, -1)
        # elif len(approx) == 3:
        #     print("triangle")
        #     cv2.drawContours(frame, [cnt], 0, (0, 255, 0), -1)
        # elif len(approx) == 4:
        #     print("square")
        #     cv2.drawContours(frame, [cnt], 0, (0, 0, 255), -1)
        # elif len(approx) == 9:
        #     print("half-circle")
        #     cv2.drawContours(frame, [cnt], 0, (255, 255, 0), -1)
        # elif len(approx) > 15:
        #     print("circle")
        #     cv2.drawContours(frame, [cnt], 0, (0, 255, 255), -1)
    
    cv2.imshow('frame', frame)
    # cv2.imshow('thresh', thresh)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cv2.waitKey(0)
cv2.destroyAllWindows()
