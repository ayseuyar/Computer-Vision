import cv2
import numpy as np

#Görüntüyü yükleyin, ardından gri tonlamalı
image = cv2.imread('deneme.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#cornerHarris işlevi, dizi veri türünün float32 olmasını gerektirir
gray = np.float32(gray)

harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)


#Köşe noktalarını büyütmek için genişletme kullanıyoruz
kernel = np.ones((7,7),np.uint8)
harris_corners = cv2.dilate(harris_corners, kernel, iterations = 2)

#Optimal bir değer için eşik, görüntüye bağlı olarak değişebilir.
image[harris_corners > 0.025 * harris_corners.max() ] = [255, 127, 127]


cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
