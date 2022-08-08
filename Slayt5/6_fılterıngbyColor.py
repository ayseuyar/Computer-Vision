import cv2
import numpy as np

#Web kamerasını başlat
cap = cv2.VideoCapture(0)

#HSV'de mavi renk aralığını tanımlayın
lower_purple = np.array([125,0,0])
upper_purple = np.array([175,255,255])

#break ifadesi çalıştırılana kadar döngü
while True:

    #web kamerası görüntüsünü oku
    ret, frame = cap.read()

    #kolayca filtreleyebilmemiz için görüntüyü RBG/BGR'den HSV'ye dönüştürün
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #yalnızca alt ve üst mavi arasındaki değerleri yakalamak için inRange kullanın
    mask = cv2.inRange(hsv_img, lower_purple, upper_purple)

    #bitsel VE maskede ve orijinal çerçevemizde gerçekleştirin
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)

    cv2.imshow('mask', mask)
    cv2.imshow('Filtered Color Only', res)
    if cv2.waitKey(1)== 13: #13 giriş tuşudur
        break
cap.release()
cv2.destoryAllWindows()

