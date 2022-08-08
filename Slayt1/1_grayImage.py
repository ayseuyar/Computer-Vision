import numpy as np
import cv2

# dizinde deneme.jpeg bulunmakta  
image = cv2.imread('deneme.jpeg')
print(image)

# dizinde deneme2.jpeg bulunmamakta fakat hata donmez  
image2 = cv2.imread('deneme2.jpeg')
# print edildiginde image2 none olarak donecektir.
print(image2)

# image'i siyah beyaz aralıgına (gray-scale) donustur 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gray-scale resmi kaydet
cv2.imwrite("denemeGray.jpeg",gray)

# Orijinal resim
cv2.imshow('Original image',image)
# Gray-scale resim
cv2.imshow('Gray image', gray)

# Komut bekle
cv2.waitKey(0)
# Butun resimleri kapat
cv2.destroyAllWindows()

