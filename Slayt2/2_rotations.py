import cv2
import numpy as np

image = cv2.imread("deneme.jpeg") # Resim Oku
height, width = image.shape[:2]# Resim Boyutlarına Bak

# Resmi ortasindan dondur
rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2),145,1)

#islemi gerceklestir
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

cv2.imwrite('rotated.jpeg', rotated_image)
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
