import cv2
import numpy as np

# NOT:
# Burada kullanılan SURF algoritması opencv gelistiricilerinin
# patentli algoritmasidir. Opencv-contrib-python kutuphanesi ile
# beraber gelmektedir. Guncel versiyonlarda (Haziran 2021 itibariyle)
# bu algoritma para ile satılmakta ve ucretsiz kutuphaneden kaldırılmıstır.
# Bu kodu calistirdiginizda bu problem ile karsilasabilirsiniz.

image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Original Image', image)
cv2.waitKey(0)


#SURF özellik algılayıcı nesnesi oluştur
surf = cv2.xfeatures2d.SURF_create(400)

#Yalnızca hession, hessionThreshold'dan daha büyük olan özellik dedektör tarafından korunur
surf.hessianThreshold = 7500
keypoints, descriptors = surf.detectAndCompute(gray, None)
print("Number of keypoints Detected: ", len(keypoints))


#Giriş görüntüsünde zengin anahtar noktalar çizin
image = cv2.drawKeypoints(image, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


cv2.imshow('Feature Method - SURF', image)
cv2.waitKey(0)


