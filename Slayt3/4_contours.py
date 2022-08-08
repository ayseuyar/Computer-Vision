import cv2
import numpy as np


#3 siyah kare ile basit bir görüntü yükleyelim
image = cv2.imread('deneme.jpeg')
cv2.imshow('deneme', image)
cv2.waitKey(0)


#gri tonlamalı
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Canny kenarlarını bul
edged =  cv2.Canny(gray, 30, 200)
cv2.imshow('Canny Edges', edged)
cv2.waitKey(0)


#Konturları Bulma
#Resminizin bir kopyasını kullanın, ör. edged.copy(), findContours görüntüyü değiştirir
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Canny Edges After Contouring', edged)
cv2.waitKey(0)

print("Number of Contours found = " +str(len(contours)))

#Tüm konturları çizin
#Tümünü çizmek için 3. parametre olarak '-1' kullanın
cv2.drawContours(image, contours, -1,(0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)

