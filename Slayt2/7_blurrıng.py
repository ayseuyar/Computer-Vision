import cv2
import numpy as np

image =cv2.imread('deneme.jpeg')
cv2.waitKey(0)

#3x3 çekirdeğimizi oluşturuyoruz
kernel_3x3 =np.ones((3, 3), np.float32)/ 9

#cv2.filter2D'yi bir görüntü ile bir çekirdeği sarmak için kullanıyoruz
blurred = cv2.filter2D(image, -1, kernel_3x3)
cv2.imshow('3x3 Kernel Blurring' , blurred)
cv2.waitKey(0)

#7x7 çekirdeğimizi oluşturuyoruz
kernel_7x7 =np.ones((7,7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
cv2.imshow('7x7 Kerne Blurring', blurred2)
cv2.waitKey(0)
