import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

template = cv2.imread("template-close.jpg", 0)
w, h = template.shape[::-1]

#video_capture.read()
while True:
    # Capture frame-by-frame
    ret, im = video_capture.read()
    imcpy = im.copy()
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(imgray, 100, 200)
    ret,thresh = cv2.threshold(edges,127,255,0)
    img, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, width, height = cv2.boundingRect(c)
        roi = im[y:y+height, x:x+width]

        # cv2.drawContours(im,contours,-1,(0,255,0),3)
        roi_cpy = im.copy()

        res = cv2.matchTemplate(imgray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(roi_cpy, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

        # Display the resulting frames
        cv2.imshow('contour', im)
        cv2.imshow('roi', roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
