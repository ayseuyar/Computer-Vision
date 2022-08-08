'''
Not: Neden oldugu bilinmiyor ama openCv detector.detect(image) satırında
C++ exception hatası veriyor. Benzer durum stackoverflow'da sorulmuş. Sorunun
muhtemel opencv versiyonunda bir bug oldugu dusunuluyor.
'''





#standart ithalat
import cv2
import numpy as np;


#resmi oku
image = cv2.imread("deneme.jpeg", cv2.IMREAD_GRAYSCALE)

#Dedektörü varsayılan parametrelerle ayarlayın
detector = cv2.SimpleBlobDetector()

#Blobları algıla
keypoints = detector.detect(image)

#Algılanan lekeleri kırmızı daireler olarak çizin
#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#daire blobun boyutuna karşılık gelir
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

#Anahtar noktaları göster
cv2.imshow("Blobs", blobs)
cv2.waitKey(0)



