import cv2
import numpy as np

print("Lutfen arkaplan yontemini seciniz:")
print("1:BackgroundSubtractorCNT")
print("2:BackgroundSubtractorGMG")
print("3:BackgroundSubtractorGSOC")
print("4:BackgroundSubtractorLSBP")
print("5:BackgroundSubtractorMOG")
while True:
    try:
        n = int(input())
        if ((n>=1) and (n<=5)):
            break
        else:
            print("Lutfen 1 ile 5 arasi bir tamsayi giriniz")
    except ValueError:
        print("Lutfen 1 ile 5 arasi bir tamsayi giriniz")
#Initialize background substractor
if n==1:
    print("Background subtraction based on counting")
    foreground_background = cv2.bgsegm.createBackgroundSubtractorCNT()
elif n==2:
    print("Background subtraction GMG")
    foreground_background = cv2.bgsegm.createBackgroundSubtractorGMG()
elif n==3:
    print("Background subtraction GSOC")
    foreground_background = cv2.bgsegm.createBackgroundSubtractorGSOC()
elif n==4:
    print("Background subtraction LSBP")
    foreground_background = cv2.bgsegm.createBackgroundSubtractorLSBP()
elif n==5:
    print("Background subtraction MOG")
    foreground_background = cv2.bgsegm.createBackgroundSubtractorMOG()
else:
    print("Bulunamadi.")
print("Videoyu kapatmak icin \"enter\"a basiniz")
cap = cv2.VideoCapture(0)
#Apply background subtractor to get our  foreground mask
while True:
    ret, frame = cap.read()
    foreground_mask = foreground_background.apply(frame)
    res = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    cv2.imshow('Output', res)
    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()

