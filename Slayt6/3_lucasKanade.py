import cv2
import numpy as np

# Load video stream
cap = cv2.VideoCapture(0)
#set parameters for ShiTomasi corner detection
feature_params = dict(maxCorners = 100,
                      qualityLevel = 0.3,
                      minDistance = 7,
                      blockSize = 7)
#Set parameters for Lucas kanade optical flow
lucas_kanade_params = dict(winSize = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#Create some random colors
color = np.random.randint(0,255,(100,3))
#Take first frame and find corners in it
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#Find inital corner locations
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask =None, **feature_params)
#Create a mask image for drawing purposes
mask = np.zeros_like(prev_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #calculate optical flow
    new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, prev_corners, None, **lucas_kanade_params)
    #select and store good points
    good_new = new_corners[status==1]
    good_old = prev_corners[status==1]
    good_new = good_new.astype(int)
    good_old = good_old.astype(int)
    #Draw the tracks
    for i in range(len(good_new[:,0])):
        new = good_new[i]
        old = good_old[i]
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a,b), 5, color[i].tolist(), -1)
    img = cv2.add(frame,mask)
    #show optical flow
    cv2.imshow('Optical Flow - Lucas-Kanade',img)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
    #Now update the previous frame and previous points
    prev_gray = frame_gray.copy()
    prev_corners = new_corners
cv2.destroyAllWindows()
cap.release()



                        
