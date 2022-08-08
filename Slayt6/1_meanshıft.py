import cv2
import numpy as np

#ınitialsize webcam
cap = cv2.VideoCapture(0)
#take first frame of the video
ret, frame = cap.read()
#setup default Location of window
r, h, c, w = 240, 100, 400, 160
track_window = (c, r, w, h)

#crop region of interest for tracking
roi = frame[r:r+h, c:c+w]

#convert cropped window to HSV color space
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#create a mask between the HSV bounds,
lower_purple = np.array([125,0,0])
upper_purple = np.array([175,255,255])
mask = cv2.inRange(hsv_roi, lower_purple, upper_purple)

#obtain the color histogram of the ROI
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])

#Normalize value to lie between the range 0, 255
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

#setup the termination criteria
#we stop calculating the centroid shift after ten iterations
#or if the centrid has moved at least 1 pixel
#term_crit = ( cv2.TERM_CRITERTA_EPS Düz dik çizgi cv2.TERM_CRITERIA_COUNT, 10, 1)
term_crit = ( cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    #read webcam frame
    ret, frame = cap.read()
    if ret == True:
        #convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Calculate the histogram back projection
        #each pixel's value is it's probability
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],0)
        #Apply meanshift to get the new locattion
        ret,track_window = cv2.meanShift(dst, track_window, term_crit)
        #Draw it on image
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w, y+h), 255, 2)
        cv2.imshow('Meansift Tracking', img2)
        if  cv2.waitKey(1) ==13: #13 is the enter key
            break
        else:
            continue
cv2.destroyAllWindows()
cap.release()

 
