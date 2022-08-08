import cv2
import numpy as np

image =cv2.imread('deneme.jpeg')
cv2.imshow('Original' , image)


#matristeki değerler toplamı 1'dir
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1,-1,-1]])

#giriş görüntüsüne farklı çekirdekler uygulamak
sharpened = cv2.filter2D(image, -1, kernel_sharpening)

cv2.imshow('Image Sharpening', sharpened)
cv2.waitKey(0)
