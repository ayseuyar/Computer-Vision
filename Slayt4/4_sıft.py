import cv2
import numpy as np

image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

#SIFT Özellik algılayıcı nesnesi oluştur
sift = cv2.SIFT_create()

#Dedektör kilit noktaları
keypoints = sift.detect(gray, None)
print("Number of keypoints Detected: " , len(keypoints))

#giriş görüntüsünde zengin anahtar noktalar çizin
cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Feature Method - SIFT', image)
cv2.waitKey(0)



