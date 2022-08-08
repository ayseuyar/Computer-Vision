import cv2
import numpy as np

#yeni imajımızı yükle
image = cv2.imread('deneme.jpeg', 0)

cv2.imshow('Original' , image)
cv2.waitKey(0)


#127'nin altındaki değerler 0'a (siyah, yukarıdaki her şey 255'e (beyaz) gider
ret,thresh1 = cv2.threshold(image,127, 255, cv2.THRESH_BINARY)
cv2.imshow('Threshold Binary' , thresh1)
cv2.waitKey(0)

#Gürültüyü ortadan kaldırdığı için görüntüleri bulanıklaştırmak için iyi bir uygulamadır.
image = cv2.GaussianBlur(image, (3,3), 0)


#AdaptiveThreshold'u kullanma
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3,5)
cv2.imshow("Adaptive Mean Thresholding", thresh)
cv2.waitKey(0)

_, th2 = cv2.threshold(image, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu's Thresholding", thresh)
cv2.waitKey(0)

#Gauss filtrelemesinden sonra Otsu'nun eşik değeri
blur = cv2.GaussianBlur(image, (5,5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Gaussian Otsu's Thresholding", thresh)
cv2.waitKey(0)
