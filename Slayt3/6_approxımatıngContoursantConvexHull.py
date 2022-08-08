import cv2
import numpy as np


#Resmi yükleyin ve bir kopyasını saklayın
image = cv2.imread('deneme.jpeg')
orig_image = image.copy()
cv2.imshow('0 - Original Image', image)
cv2.waitKey(0)

#Gri tonlamalı  ve ikili hale getirme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

#Kontur bul
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#Her konturu yineleyin ve sınırlayıcı dikdörtgeni hesaplayın
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x,y),(x+w,y+h), (0,0,255),2)
    cv2.imshow('Bounding Rectangle', orig_image)

cv2.waitKey(0)

#Her konturu yineleyin ve yaklaşık kkonturu hesaplayın
for c in contours:
    #Doğruluğu kontur çevresinin yüzdesi olarak hesaplayın
    accuracy = 0.03 * cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow('Approx Poly DP', image)

cv2.waitKey(0)    
