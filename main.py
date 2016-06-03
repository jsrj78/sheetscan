import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

#video_capture.read()
while True:
    # Capture frame-by-frame
    ret, im = video_capture.read()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(imgray, 100, 200)
    ret,thresh = cv2.threshold(edges,127,255,0)
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im,contours,-1,(0,255,0),3)

    # Display the resulting frames
    cv2.imshow('out', im)
    cv2.imshow('Canny', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
