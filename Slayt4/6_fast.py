import cv2
import numpy as np


image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)


#HIZLI özellik algılayıcı nesnesi oluştur
fast = cv2.xfeatures2d.StarDetector_create()


#Anahtar noktaları edinin, varsayılan olarak maksimum olmayan bastırma açıktır
#set fast.setBool('nonmaxSuppression', False) özelliğini kapatmak için 
keypoints = fast.detect(gray, None)
print("Number of keypoints Detected: ", len(keypoints))


#Giriş görüntüsünde zengin anahtar noktalar çizin
cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('Feature Method - FAST', image)
cv2.waitKey(0)

