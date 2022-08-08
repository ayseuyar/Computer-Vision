import cv2
import numpy as np

image = cv2.imread('deneme.jpeg', cv2.IMREAD_GRAYSCALE)


cv2.imshow('Original', image)
cv2.waitKey(0)

#Çekirdek boyutumuzu tanımlayalım
kernel = np.ones((15, 15), np.uint8)

#şimdi erozyona uğradık
erosion =cv2.erode(image, kernel, iterations = 1)
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)


#
dilation = cv2.dilate(image, kernel, iterations =1)
cv2.waitKey(0)

#Opening - Gürültüyü gidermek için
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
cv2.imshow('Opening', opening)
cv2.waitKey(0)

#Closing - Gürültüyü gidermek için
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
