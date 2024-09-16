import cv2 as cv
import numpy as np

# Create a blank image (500x500 pixels, 8-bit unsigned integers, black)
blank = np.zeros((500, 500, 3), dtype='uint8')
blank[240:500,240:250] = [255, 0, 255] #Blue, Green, Red
cv.imshow('Color', blank)
#rectangle
cv.rectangle(blank, (50,50), (550,550), (0, 255, 0), thickness=5)
cv.imshow('New Color', blank)
cv.waitKey(0)
cv.destroyAllWindows()
    