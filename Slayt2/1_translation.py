import cv2
import numpy as np

image = cv2.imread("deneme.jpeg") # Resmi Oku
height, width = image.shape[:2] # Resmin Boyutlarına bak
quarter_height, quarter_width = height/4, width/4 # Kaydırma Miktarı
T = np.float32([[1, 0,quarter_width], [0, 1, quarter_height]]) # Transformation Matrix
image2 = cv2.warpAffine(image, T, (width, height)) # WartAffine
cv2.imwrite("deneme2.jpeg",image2) # Resmi Kaydet
cv2.imshow("Orijinal Resim",image) # Orjinal Resim
cv2.waitKey(0) # Kullanıcıdan tepki bekle
cv2.imshow("Modifiye Resim",image2) # Modifiye Resim
cv2.waitKey(0) # Kullanıcıdan tepki bekle
