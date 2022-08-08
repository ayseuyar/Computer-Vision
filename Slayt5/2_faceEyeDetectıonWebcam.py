import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

def face_detector(img,  size=0.5):
    #görüntüyü gri tonlamaya dönüştür
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) is 0:
        print("Yuz Bulunamadi")
        return img
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
    return img

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor' , face_detector(frame))
    if cv2.waitKey(1) == 13: #13 Giriş Tuşudur
        break
cap.release()
cv2.destroyAllWindow()
 
