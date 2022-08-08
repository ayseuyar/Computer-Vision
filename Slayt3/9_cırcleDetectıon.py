import cv2
import numpy as np
import cv2 as cv
 
image = cv2.imread('deneme.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
blur = cv2.medianBlur(gray, 5)
circles = cv2.HoughCircles(blur, cv.HOUGH_GRADIENT, 1.5, 10,
                           param1=100, param2=30,
                           minRadius=50, maxRadius=60)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # dış daireyi çiz
        cv2.circle(image,(i[0], i[1]), i[2], (255, 0, 0), 2)
      
        # dairenin merkezini çiz
        cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)
 
cv2.imshow('detected circles', image)
cv2.waitKey(0)

