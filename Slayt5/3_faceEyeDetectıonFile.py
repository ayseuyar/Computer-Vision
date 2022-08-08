import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

#Resmimizi yükleyin ve ardından gri tonlamaya dönüştürün
img = cv2.imread('lenna.png')
cv2.imshow("Orjinal Resim",img)
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)
if len(faces) is 0:
    print("Yuz Bulunamadi")
else:
    print("Bulunan Yuz Sayisi:"+str(len(faces)))
    for (x,y,w,h) in faces:
        x = x - 25
        w = w + 50
        y = y - 25
        h = h + 50
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
    cv2.imshow("Yuz ve Goz",img)
    cv2.waitKey(0)
cv2.destroyAllWindow()
 
