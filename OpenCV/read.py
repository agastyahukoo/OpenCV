import cv2 as cv

img = cv.imread('Images/20240915.jpg.jpg') #takes in path of the image and returns it as a matrix of pixels
cv.imshow('Cat', img) #First parameter is the name of the window and the second is the matrix
cv.waitKey(0) #waits for a key to be press

capture = cv.VideoCapture('Videos/video.mov') #takes integer arguments for webcam or a path to a video file
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF==('d'):
        break
capture.release
cv.destroyAllWindows()

def rescaleFrame(frame, scale=0.75):
    width = frame.shape[1] * scale
    height = frame.shape[0] * scale
    dimensions= (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

cv.waitKey(0)


#frame_resized = rescaleFrame(frame)