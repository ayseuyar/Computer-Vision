import cv2
import numpy as np

image = cv2.imread('deneme.jpeg')

#Birlerden oluşan bir matris oluşturun, ardından bunu 100'lük bir ölçekleyici ile çarpın
#Bu, tüm değerleri 100 olan görüntümüzün bazı boyutlarına sahip bir matris verir.
M = np.ones(image.shape, dtype = "uint8") * 125

#Bunu, bu M matrisini imajımıza eklemek için kullanırız.
#Parlaklık artıına dikkat edin
added = cv2.add(image, M)
cv2.imshow("Added" , added)

#Aynı şekilde  çıkarma da yapabiliriz
#Parlaklıktaki azalmaya dikkat edin
subtracted = cv2.subtract(image, M)
cv2.imshow("subtracted" , subtracted)

cv2.waitKey(0)
