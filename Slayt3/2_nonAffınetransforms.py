import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('deneme.jpeg')

cv2.imshow('Original' , image)
cv2.waitKey(0)

points_A = np.float32([[320,15], [700,215], [85, 610], [530, 780]])


points_B = np.float32([[80, 0], [420, 0], [0, 594], [420, 594]])


M = cv2.getPerspectiveTransform(points_A, points_B)

warped = cv2.warpPerspective(image, M, (420, 594))

cv2.imshow('warpRespective', warped)
cv2.waitKey(0)
