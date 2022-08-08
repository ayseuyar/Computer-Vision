import numpy as np
import cv2

#OpenCV'nin CascadeClassifier işlevini,
#sınıflandırıcı (XML dosya formatı) başlatıldı
face_classifier = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

#Resmimizi yükleyin ve ardından gri tonlamaya dönüştürün
image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#sınıflandırıcımız, algılanan yüzün ROI'sini bir demet olarak döndürür
#Sol üst koordinata ve sağ alt koordinatlara bakar
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

#yüz algılanmadığında, face_classifier döner ve boş demet
if faces is ():
    print("No faces found")


#Yüz dizimizi yineliyoruz ve bir dikdörtgen çiziyoruz
#yüzlerde her yüzün üzerinde
for (x,y,w,h) in faces:
    cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)

 
