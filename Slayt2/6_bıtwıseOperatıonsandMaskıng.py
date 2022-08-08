import cv2
import numpy as np

#Neden sadece iki boyutlu olduğunu merak ediyorsanız, bu gri tonlamalı bir görüntü
#Renkli bir görüntü yaparsak, kullanırdık
#rectantle = np.zeros((300, 300, 3), np.uint8)

#Kare yapmak
square = np.zeros((300, 300),np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
cv2.imshow("Square", square)
cv2.waitKey(0)

#Elips yapmak
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (120,40), 180, 0, 360,(255,0,0),5)
cv2.imshow("Ellipse", ellipse)
cv2.waitKey(0)


#Yalnızca kesiştikleri yeri gösterir
And = cv2.bitwise_and(square, ellipse)
cv2.imshow("AND", And)
cv2.waitKey(0)

#Kare veya elipsin nerede olduğunu gösterir
bitwiseOr = cv2.bitwise_or(square, ellipse)
cv2.imshow("OR", bitwiseOr)
cv2.waitKey(0)


#Her ikisinin de kendi başına nerede her şeyi gösterir
bitwiseXor = cv2.bitwise_xor(square, ellipse)
cv2.imshow("XOR", bitwiseXor)
cv2.waitKey(0)

#Meydanın parçası olmayan her şeyi gösterir
bitwiseNot_sq = cv2.bitwise_not(square)
cv2.imshow("NOT - square",bitwiseNot_sq)
cv2.waitKey(0)


### Son işlemin görüntüyü tamamen tersine çevirdiğine dikkat edin
