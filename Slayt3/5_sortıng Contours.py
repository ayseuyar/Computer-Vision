import cv2
import numpy as np


#Şekil şablonunu veya referans görüntüsünü yükleyin
template = cv2.imread('deneme.jpeg', 0)
cv2.imshow('Template', template)
cv2.waitKey()

#Hedef görüntüyü eşleştirmeye çalıştığımız şekillerle yükleyin
target = cv2.imread('deneme.jpeg')
target_gray = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)

#cv2.findConstou'yu kullanmadan önce her iki görüntüyü de eşik tutun
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

#Şablonda konturu bul
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#En büyüğünü kaldırabilmemiz için konturları alana göre sıralamamız gerekiyor.
#görüntü anahattı olan kontur
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

#İkinci hedef görüntüden konturları çıkarıyoruz
template_contour = contours[1]

#İkinci hedef görüntüden konturları çıkarın
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    #Hedef görüntüdeki her konturu yineleyin ve
    #kontur şeklini karşılaştırmak için cv2.matchShape kullanın
    match = cv2.matchShapes(template_contour, c, 1, 0.0)
    print(match)
    #eşleşme değeri 0.15'ten küçükse
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour =[]

cv2.drawContours(target, contours, -1, (0,255,0), 3)
cv2.imshow('Output', target)
cv2.waitKey()



