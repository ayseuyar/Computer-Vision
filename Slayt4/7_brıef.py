import cv2
import numpy as np

# Bu fonksiyonlar icin opencv-contrib gereklidir. Yuklemek icin:
# ./pip.exe install opencv-python opencv-contrib-python
# Eger problem hala devam ediyorsa opencv ve opencv-contrib'i kaldirip
# tekrardan deneyin
# ./pip.exe uninstall opencv-python opencv-contrib-python,
# ./pip.exe install opencv-python opencv-contrib-python

image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)


#HIZLI özellik algılayıcı nesnesi oluştur
fast = cv2.xfeatures2d.StarDetector_create()

#KISA çıkarıcı nesnesi oluştur
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

#Önemli noktaları belirleyin
keypoints = fast.detect(gray, None)

#BRIEF kullanarak tanımlayıcıları ve yeni son anahtar noktaları edinin
keypoints, descriptors = brief.compute(gray, keypoints)
print = "Number of keypoints Detecte: ", len(keypoints)

#Giriş görüntüsünde zengin anahtar noktalar çizin
cv2.drawKeypoints(image, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('Feature Method - BRIEF', image)
cv2.waitKey(0)

