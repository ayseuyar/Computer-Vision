import cv2
import numpy as np

image = cv2.imread("deneme.jpeg") # Resim oku
height, width=image.shape[:2] #Resim boyutlarına bak

image_scaled =cv2.resize(image, None, fx=0.75, fy=0.75)
cv2.imshow('Scaling - linear Interpolation' , image_scaled)
cv2.waitKey(0)

img_scaled = cv2.resize(image, None, fx=2, fy=2,interpolation = cv2.INTER_CUBIC)
cv2.imshow('Scaling - cubic Interpolation' , img_scaled)

img_scaled = cv2.resize(image, (900, 400), interpolation = cv2.INTER_AREA)
cv2.imshow('Scaling - Skewed Size' , img_scaled)

cv2.imwrite("deneme1.jpeg",img_scaled) #Resmi Kaydet
cv2.imshow("Orjinal Resim",image) #Orjinal Resim
cv2.waitKey(0) #Kullanıcıdan Tepki Bekle
cv2.imshow("Modifiye Resim",img_scaled) #Modifiye Resim
cv2.waitKey(0) # Kullanıcıdan Tepki Bekleme
