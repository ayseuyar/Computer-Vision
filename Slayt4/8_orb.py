import cv2
import numpy as np


image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

#ORB nesnesi oluştur, istediğimiz anahtar nokta sayısını belirtebiliriz
orb = cv2.ORB_create(50)

#Önemli noktaları belirleyin
keypoints = orb.detect(gray, None)

#tanımlayıcıları elde etmek
keypoints, descriptors = orb.compute(gray, keypoints)
print = "Number of keypoints Detecte: ", len(keypoints)

#Giriş görüntüsünde zengin anahtar noktalar çizin
cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('Feature Method - ORB', image)
cv2.waitKey(0)
