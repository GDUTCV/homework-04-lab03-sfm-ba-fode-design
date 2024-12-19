import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)


ret, frame1 = cap.read()
if not ret:
    print("无法打开摄像头")
    exit()


prev_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)


hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())


while True:
    
    ret, frame2 = cap.read()
    if not ret:
        break
    
    
    next_gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    
    flow = cv.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  
    hsv[..., 0] = ang * 180 / np.pi / 2  
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  

    
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    
    cv.imshow('Optical Flow', bgr)
    
    
    prev_gray = next_gray.copy()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()