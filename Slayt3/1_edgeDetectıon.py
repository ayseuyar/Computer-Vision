import cv2
import numpy as np


image = cv2.imread('deneme.jpeg',0)

height, width = image.shape

#Sobel kenarlarını çıkarın
sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

cv2.imshow('Original' , image)
cv2.waitKey(0)
cv2.imshow('Sobel X' , sobel_x)
cv2.waitKey(0)
cv2.imshow('Sobel_Y' , sobel_y)
cv2.waitKey(0)

sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
cv2.imshow('sobel_OR' , sobel_OR)
cv2.waitKey(0)

laplacian = cv2.Laplacian(image, cv2.CV_64F)
cv2.imshow('Laplacian' , laplacian)
cv2.waitKey(0)

canny = cv2.Canny(image, 20, 170)
cv2.imshow('Canny' , canny)
cv2.waitKey(0)
