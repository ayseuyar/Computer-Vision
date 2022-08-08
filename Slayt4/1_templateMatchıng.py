import cv2
import numpy as np

#Giriş görüntüsünü yükleyin ve gri tonlamaya dönüştürün
image = cv2.imread('deneme.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Orijinal Resim",gray)
cv2.waitKey(0)

#Şablon resmini yükle
template = cv2.imread('p.png', 0)
height = len(template[:,0])
width = len(template[0,:])
cv2.imshow("Referans Resim",template)
cv2.waitKey(0)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc =cv2.minMaxLoc(result)

#Sınırlayıcı kutu oluştur
top_left = max_loc
bottom_right =  (top_left[0] + width, top_left[1] + height)
cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

cv2.imshow('Bulunan Referans', image)
cv2.waitKey(0)


