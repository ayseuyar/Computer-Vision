import cv2
import numpy as np

image = cv2.imread('cerceve1.jpg')

#Gri Tonlamalı ve Canny Edges ayıklandı
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200, None, 3)


#1 piksellik bir rho doğruluğu kullanarak HoughLines'ı çalıştırın
#1 derece olan np.pi/180'in teta doğruluğu
#Çizgi eşiğimiz 150 olarak ayarlanmıştır (çizgideki nokta sayısı)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150, None, 0, 0)


#Her Satır boyunca yinelenir ve onu biçime dönüş
#cv.Lines tarafından gerekli (yani bitiş noktaları gerektiren)
if lines is not None:
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            xteta = a * rho
            yteta = b * rho
            x1 = int(xteta + 1000 * (-b))
            y1 = int(yteta + 1000 * (a))
            x2 = int(xteta - 1000 * (-b))
            y2 = int(yteta - 1000 * (a))
            cv2.line(image, (x1,y1), (x2,y2), (255,150,150), 10)


cv2.imshow('Hough Lines', image)
cv2.waitKey(0)


