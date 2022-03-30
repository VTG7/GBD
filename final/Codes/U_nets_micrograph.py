from tqdm import tqdm
import random
import numpy as np
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import cv2
import glob
#import countour_detection
#from PIL import Image




import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
TRAIN_PATH= r'D:/Videh_Acads/Machine_Learning/data/Sir_Micrographs/micrographs Inconel-20200827T122844Z-001/Train_Images/'
TRUTH_PATH = r'D:/Videh_Acads/Machine_Learning/data/Sir_Micrographs/micrographs Inconel-20200827T122844Z-001/Ground_Truth/'
seed = 42
np.random.seed = seed

#IMG_HEIGHT = 572
#IMG_WIDTH = 572
IMG_HEIGHT = 1084
IMG_WIDTH = 1084
IMG_CHANNELS = 3

#TRUTH_HEIGHT = 388
#TRUTH_WIDTH = 388
TRUTH_HEIGHT = 900
TRUTH_WIDTH = 900

train_ids = next(os.walk(TRAIN_PATH))[2]
truth_ids = next(os.walk(TRUTH_PATH))[2]

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
Y_train = np.zeros((len(truth_ids), TRUTH_HEIGHT , TRUTH_WIDTH, 1), dtype=np.uint8)


'''print('Resizing training images')
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
    path = TRAIN_PATH + id_
    img = imread(path)
    print(path)
    #print(img.shape) #[:,:,:IMG_CHANNELS]  
    img = (resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True , mode='constant', preserve_range=True)).astype(np.uint8)
    #print(img.shape)
    #print(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(img.shape)
    img = img[:, :, np.newaxis]
    #print(img.shape)
    X_train[n] = img
    
    
    
    
    
     #Fill empty X_train with values from img
    
    
              
     
print('Resizing ground truth images') 
for n, id_ in tqdm(enumerate(truth_ids), total=len(truth_ids)):
    path = TRUTH_PATH + id_
    #print(path)
    img = imread(path)
    print(path)
    if n == 28 or n==29 or n == 30:
        img = img[:,:,0]
    #shape = str(img.shape)
    #nos = str(n)
    #print(nos+' '+ shape)    
    #print(img)
    #print(img.shape)
    #cv2.imshow('image', img)  
    #img = img*255  
    #print(img)
    img = (resize(img, (TRUTH_HEIGHT, TRUTH_WIDTH), anti_aliasing=True , mode='constant', preserve_range=True)).astype(np.uint8)
    #print(img.shape)
    img = (img>153).astype(np.uint8)
    img = img[:, :, np.newaxis]
    #print(img.shape)
    #img = img.astype(np.uint8)
    Y_train[n] = img
    #print(Y_train[0])'''
    
    
#import os
print('Resizing training images')
directory = TRAIN_PATH
files = sorted(glob.glob(TRAIN_PATH + '*.jpg'))
n =0
for filename in files:
    if filename.endswith(".jpg"):
      #print(filename)
      img = imread(filename)
      img = (resize(img, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True , mode='constant', preserve_range=True)).astype(np.uint8)
      img = img[:, :, np.newaxis]
      X_train[n] = img
      n = n+1
      continue
    else:
        continue 

print('Resizing truth images')
directory = TRUTH_PATH
files = sorted(glob.glob(TRUTH_PATH + '*.jpg'))
n = 0
for filename in files:
    if filename.endswith(".jpg"):
      #print(filename)
      img = imread(filename)
      if img.ndim == 3:
          img = img[:,:,0]
      img = (resize(img, (TRUTH_HEIGHT, TRUTH_WIDTH), anti_aliasing=True , mode='constant', preserve_range=True)).astype(np.uint8)
      img = (img>153).astype(np.uint8)
      img = img[:, :, np.newaxis]
      Y_train[n] = img
      n = n+1
      continue
    else:
        continue     
    
    
    
    
    

print('Done!')



'''fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
#print(X_train[0])
imgplot = plt.imshow(X_train[20], cmap='gray')
ax.set_title('Before')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 2, 2)
#print(Y_train[0])
imgplot = imshow(Y_train[20], cmap='gray', vmin=0, vmax=1)
#imgplot.set_clim(0.0, 0.7)
ax.set_title('After')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
plt.show()'''




'''image_x = random.randint(0, len(train_ids))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()'''
#Y_train = Y_train/255.0


'''inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, 1))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid')(c5)
c4 = tf.image.resize(c4, (56, 56))
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid')(c6)
c3 = tf.image.resize(c3, (104, 104))
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid')(c7)
c2 = tf.image.resize(c2, (200, 200))
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(c8)
c1 = tf.image.resize(c1, (392, 392))
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


results = model.fit(X_train, Y_train, validation_split=0.1, epochs=3, batch_size= 5)     #Pls note

#dx = random.randint(0, len(X_train))'''

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, 1))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='valid')(c5)
c4 = tf.image.resize(c4, (120, 120))
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='valid')(c6)
c3 = tf.image.resize(c3, (232, 232))
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='valid')(c7)
c2 = tf.image.resize(c2, (456, 456))
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(c8)
c1 = tf.image.resize(c1, (904, 904))
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='valid')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


results = model.fit(X_train, Y_train, validation_split=0.2, epochs=2, batch_size= 3)


'''preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
mean1 = preds_train.mean()
mean2 = preds_val.mean()
print("Shape of preds_val")
print(preds_val.shape)
#preds_test = model.predict(X_test, verbose=1)

 
preds_train_t = 1*((preds_train > mean1).astype(np.uint8))
preds_val_t = 1*((preds_val > mean2).astype(np.uint8))
print(preds_val)
#preds_test_t = (preds_test > 0.5).astype(np.uint8)



# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(X_train[ix], cmap='gray')
ax.set_title('Before')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 3, 2)
imgplot = imshow(Y_train[ix], cmap='gray', vmin=0, vmax=1)
#imgplot.set_clim(0.0, 0.7)
ax.set_title('Ground-Truth')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 3, 3)
imgplot = imshow(preds_train_t[ix], cmap='gray', vmin=0, vmax=255)
#imgplot.set_clim(0.0, 0.7)
ax.set_title('Prediction')
plt.show()




# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(X_train[ix], cmap='gray')
ax.set_title('Before')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 3, 2)
imgplot = imshow(Y_train[ix], cmap='gray', vmin=0, vmax=1)
#imgplot.set_clim(0.0, 0.7)
ax.set_title('Ground-Truth')
#plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax = fig.add_subplot(1, 3, 3)
print(preds_val_t.shape)
imgplot = imshow(preds_val_t[0], cmap='gray', vmin=0, vmax=1)
#imgplot.set_clim(0.0, 0.7)
ax.set_title('Prediction')
plt.show()'''

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_hand_gt_big.h5")
print("Saved model to disk")

 