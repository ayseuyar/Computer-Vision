import cv2
import numpy as np


#histogram grafiklerimizi oluşturmak için matplotlib'i içe aktarmamız gerekiyor
from matplotlib import pyplot as plt


image = cv2.imread('denemeHist.png')
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])


#bir histogram çiziyoruz, ravel() görüntü dizimizi düzleştiriyor
plt.hist(image.ravel(), 256, [0, 256]); plt.show()



#ayrı renk kanallarını görüntüleme
color = ('b', 'g', 'r')

#şimdi renkleri ayırıyoruz ve her birini Histogramda çiziyoruz
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color = col)
    plt.xlim([0,256])

plt.show()    


