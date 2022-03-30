import skimage.io
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.io import imsave
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2 

import tensorflow
from tensorflow import keras
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json


IMG_HEIGHT = 1084               #1084, 572
IMG_WIDTH = 1084                #1084, 572
IMG_CHANNELS = 3



#img = mpimg.imread('1.tif')
img = imread('test_1.jpg')
#img = imread('Image_Grayscale.jpg')
img_orig = resize(img, (IMG_HEIGHT, IMG_WIDTH))

#if the image is coloured
if img.ndim == 3:
    img = (resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:,:, np.newaxis]
    orig = img.copy()
#or if its gray-scaled    
else:
    img = (resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)).astype(np.uint8) 
    img = img[:,:, np.newaxis] 
    orig = img.copy()
    
          


img = img[np.newaxis,:,:,:]
print(img.shape)

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_hand_gt_big.h5")          #model_hand_gt_big.h5           #model_countour_gabor_900.h5
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



pred = loaded_model.predict(img, verbose=1)
mean = pred.mean()
std = pred.std()
#print(mean)
#print(std)
#print(pred)
#print("mean= " + str(mean))
#result_2 = (pred > 0.7).astype(np.uint8)
#result_3 = (pred > 0.7).astype(np.uint8)
#result_4 = (pred > 0.9).astype(np.uint8)
result_1 = (pred>mean).astype(np.uint8)
result_1 = result_1*255
#result_1 = pred
#print((pred- mean))
#result_1 = result_1
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(orig, cmap='gray')
ax.set_title('original')
# ax = fig.add_subplot(1, 2, 2)
# imgplot = plt.imshow(result_1[0], cmap='gray')
# ax.set_title('pred>0.9')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
# ax = fig.add_subplot(1, , 3)
# imgplot = plt.imshow(result_2[0], cmap='gray', vmin=0, vmax=1)
# #imgplot.set_clim(0.0, 0.7)
# ax.set_title('pred>0.7')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(result_1[0], cmap='gray', vmin=0, vmax=1)
#print(result_1[0].shape)
#imga = cv2.cvtColor(result_1[0], cv2.COLOR_BGR2GRAY)
imsave('test_1_pred',result_1[0])
#imsave('pred_big_hand_meanThreshold_1.jpg', result_1[0])
#, cmap = 'gray', vmin =0, vmax =1
ax.set_title('pred')
#ax = fig.add_subplot(1, 5, 5)
#imgplot = plt.imshow(result_4[0], cmap='gray', vmin=0, vmax=1)
#ax.set_title('pred>0.9')
plt.show()
