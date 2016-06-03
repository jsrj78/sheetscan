import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3,3))

    edges = cv2.Canny(gray, 100, 200)
    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im2, contours, -1, (0,255,0), 3)

    # Display the resulting frames
    cv2.imshow('Canny', edges)
    cv2.imshow('Contours', im2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
