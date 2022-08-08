import cv2
import numpy as np

#eskiz oluşturma fonksiyonumuz
def sketch(image):
    #görüntüyü gri tonlamaya dönüştür
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Görüntüyü gri tonlamalı olarak temizle
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    #Kenarları ayıkla
    canny_edges = cv2.Canny(img_gray_blur, 10, 70)

    #Bir ters çevirme yapın, görüntüyü ikili hale getirin
    ret, mask = cv2.threshold(canny_edges, 70, 225, cv2.THRESH_BINARY_INV)
    return mask


#Web kamerasını başlat, cap, VideoCapture tarafından sağlanan nesnedir
#Başarılı olup olmadığını gösteren bir boole içerir (ret)
#Ayrıca web kamerasından (çerçeve) toplanan görüntüleri de içerir.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Our Live Sketcher' , sketch(frame))
    if cv2.waitKey(1) == 13: #3 Giriş Tuşudur
        break

