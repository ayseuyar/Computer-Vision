import cv2
import numpy as np

image = cv2.imread('deneme.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 150)

for corner in corners:
    x, y = corner[0]
    x = int(x)
    y = int(y)
    cv2.rectangle(image,(x-10,y-10),(x+10,y+10),(0,255,0), 2)

cv2.imshow("Corners Found", image)
cv2.waitKey(0)


