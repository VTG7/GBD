import cv2
import imutils
from matplotlib import pyplot as plt
from skimage.filters import threshold_sauvola #edited

#Edge detection below
image = cv2.imread('<insert path here>')
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height = 500)
# convert the image to grayscale, blur it, and find edges
# in the image

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
orig=gray.copy()
T_sauvola = threshold_sauvola(orig, 21)
orig = (orig > T_sauvola).astype("uint8") * 255
gray=orig

kernel=np.ones((3,3),np.uint8)

gray=cv2.erode(gray, kernel ,iterations=1)
gray=cv2.dilate(gray, kernel ,iterations=1)

gray = cv2.GaussianBlur(gray, (15, 15), 0) # this is the part where I tweaked the parameters to get to have contours over the entire micrograph
edged = cv2.Canny(gray, 130, 140)

res = cv2.morphologyEx(edged,cv2.MORPH_CLOSE, kernel)
invert_image=cv2.bitwise_not(res)

dst = cv2.addWeighted(orig, 0.4, invert_image, 0.6, 0.0)
dst=cv2.erode(dst, kernel ,iterations=1)
dst=cv2.dilate(dst, kernel ,iterations=1)
dst=cv2.erode(dst, kernel ,iterations=1)
dst=cv2.dilate(dst, kernel ,iterations=3) # @ Videh, try putting iterations = 2 here, the edges become more prominent, however noise increases to some extent.
plt.imshow(dst)
plt.show()

#NOTE1: The initial method of contour detection has been replaced because:
# 1) The edges were being geometricised(that is sharp edges instead of curvy ones)
# 2) There was considerable gap between the inner and outer edge of a boundary.Tried dilation, but it didn't help.
#Note2: The new method seems to have an advantage.This can only be verified via training the model on the new dataset created by this algorithm.
