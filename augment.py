# This is a slightly modified version of the kernel present here : https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
# For Videh: if you need to use this code for all images at one go, then please refer to the link above.Some changes to the below code will 
# have to be made though.Even then resize the image for faster processing(done below).

import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import imutils

# this function does the elastic transformation.As written in the UNet paper, "Especially random elastic deformations of the training samples 
# seem to be the key concept to train a segmentation network with very few annotated images."

def elastic_transform(image,alpha,sigma,alpha_affine,random_state=None):
    if random_state is None:
       random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

# Define function to draw a grid.Videh, please remove this piece when you use it.This code is for illustrating the warping transformation visually.
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(1,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(1,))

# Load images
im = cv2.imread('<insert image path>')

im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_mask = cv2.imread('<insert mask path>')
im=imutils.resize(im,height=500).  # v imp. for faster processing.Don't remove the resizing step.One may resize it again to original size if required later. 

# Draw grid lines
draw_grid(im, 50)

im_aug = elastic_transform(im, im.shape[1] * 2, im.shape[1] * 0.08, im.shape[1] * 0.08)
im_mask_aug=elastic_transform(im_mask, im_mask.shape[1] * 2, im_mask.shape[1] * 0.08, im_mask.shape[1] * 0.08)
plt.imshow(im_aug,cmap='gray')
plt.imshow(im_mask_aug,cmap ='gray')
