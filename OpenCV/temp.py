import cv2 as cv

capture = cv.VideoCapture('Videos/video.mov') #takes integer arguments for webcam or a path to a video file
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release
cv.destroyAllWindows()

