import cv2
import numpy as np
import matplotlib.pyplot as plt


#Görüntüyü yükleyin, ardından gri tonlamalı
image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# orijinal resmi göster
cv2.imshow('Input Image', image)

#piksel cinsinden h x w
cell_size = (8, 8)

#hücrelerde h x w
block_size = (2, 2)

#yönlendirme kutuları sayısı
nbins = 9


#OpenCV'nin HOG Tanımlayıcısını kullanarak
#winSize, hücre boyutunun katlarına kırpılan görüntünün boyutudur
hog = cv2.HOGDescriptor(_winSize=(gray.shape[1] // cell_size[1]*cell_size[1],
                                  gray.shape[0]//cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1]*cell_size[1],
                                     block_size[0]*cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

#hog_feats oluşturmak için kullandığımız numpy dizi şekli oluşturun
n_cells = (gray.shape[0] // cell_size[0], gray.shape[1] // cell_size[1])

#önce blokları satırlara göre indeksliyoruz.
#hog_feats artık her yönde gradyan genliklerini içerir,
#her grup için kendi grubunun her hücresi için. İndeksleme satırlar, ardından sütunlar tarafından yapılır.
hog_feats = hog.compute(gray).reshape(n_cells[1] - block_size[1] +1,
                                       n_cells[0] - block_size[0]+1,
                                       block_size[0], block_size[1], nbins).transpose((1, 0, 2, 3, 4))

#Create our gradients array with nbin dimensions to store gradient arientations
gradients = np.zeros((n_cells[0], n_cells[1], nbins))

#Create array of dimenstions
cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

#block Normalization
for off_y in range(block_size[0]):
    for off_x in range(block_size[1]):
        gradients[off_y:n_cells[0] - block_size[0] + off_y +1,
                  off_x:n_cells[1] - block_size[1] + off_x +1]+=\
            hog_feats[:, :, off_y,off_x, :]
        cell_count[off_y:n_cells[0] - block_size[0] + off_y + 1,
                   off_x:n_cells[1] - block_size[1] + off_x + 1] += 1

#Ortalama gradyanlar
gradients /= cell_count

#HOG'ları Matplotlib kullanarak çizin
#Açı 360/ nbins * yön
color_bins = 5
plt.pcolor(gradients[:, :, color_bins])
plt.gca().invert_yaxis()
plt.gca().set_aspect('equal' , adjustable='box')
plt.colorbar()
plt.show()
cv2.destroyAllWindows()

