import cv2
import imutils
from matplotlib import pyplot as plt
#Edge detection below
image = cv2.imread('<insert path here>')
ratio = image.shape[0] / 500.0

orig = image.copy()
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0) # this is the part where I tweaked the parameters to get to have contours over the entire micrograph
edged = cv2.Canny(gray, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:]
#looping over the entire contour

for c in cnts:
    perimeter=cv2.arcLength(c,True)
    approx_curve = cv2.approxPolyDP(c,0.02*perimeter, True)
    cv2.drawContours(image, [approx_curve], -1, (0, 255, 0), 2)

plt.imshow(image)
plt.show()

