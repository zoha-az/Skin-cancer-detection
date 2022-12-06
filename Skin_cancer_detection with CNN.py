# importing the required libraries
# Mainly, we need numpy, pandas, matplotlib and openCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from PIL import Image
from tensorflow.keras import utils

# Loading the dataset: we have 4 folders: train benign, train malignant, test benign, test malignant

folder_benign_train = './train/benign'
folder_malignant_train = './train/malignant'

folder_benign_test = './test/benign'
folder_malignant_test = './test/malignant'


read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

# Loading in train images 
ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
X_benign = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
X_malignant = np.array(ims_malignant, dtype='uint8')

# Loading in test images
ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
X_benign_test = np.array(ims_benign, dtype='uint8')
ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
X_malignant_test = np.array(ims_malignant, dtype='uint8')

# Creating the labels
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])

y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])


# Merging the data 
X_train = np.concatenate((X_benign, X_malignant), axis = 0)
y_train = np.concatenate((y_benign, y_malignant), axis = 0)

X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)


# Checking the shape of X_train, X_test, y_traim, y_test
print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Display first 15 images and how they are classified
w=40
h=30
fig=plt.figure(figsize=(12, 8))
columns = 5
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if y_train[i] == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(X_train[i], interpolation='nearest')
plt.show()

plt.bar(0, y_train[np.where(y_train == 0)].shape[0], label = 'benign')
plt.bar(1, y_train[np.where(y_train == 1)].shape[0], label = 'malignant')
plt.legend()
plt.title("Training Data")
plt.show()

plt.bar(0, y_test[np.where(y_test == 0)].shape[0], label = 'benign')
plt.bar(1, y_test[np.where(y_test == 1)].shape[0], label = 'malignant')
plt.legend()
plt.title("Test Data")
plt.show()


figure_size = 3


# Defining the preprocessing function: DHR + Filter (mean, median or non-local means + morphology)

def preprocessing(image):
  # Digital hair removal

  #color to gray-scale
  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # blackhat operator
  kernel = cv2.getStructuringElement(1,(17,17))
  blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
  # threshold image after blackhat
  _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
  # Inpainting to modify the image after removing hair/ lines
  # inputs: image, mask, Radius of a circular neighborhood of each point inpainted, Inpainting method
  dhr = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
  
  # Since the previous step gives a color image, another conversion to gray-scale
  grayscale_2 = cv2.cvtColor(dhr, cv2.COLOR_BGR2GRAY)
  
  ### Applying one of the filters below: nlm/ mean/ median


  # Non local means filter
  # inputs: image, h, search window size, block size
  # h: Big h value perfectly removes noise but also removes image details, 
  # smaller h value preserves details but also preserves some noise
  # search_window	Size: in pixels of the window that is used to compute weighted average for given pixel. Should be odd.
  # search window size: Recommended value 21
  # block_size: Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7
  h=3
  nlm = cv2.fastNlMeansDenoising(grayscale_2,None,h, 7,21)

  # Median filter
  # k: Size of the kernel.  Must be odd.
  k = 5
  # median_fil = cv2.medianBlur(grayscale_2, k)

  # Mean filter
  # k: Size of the kernel  Must be odd.
  # mean_fil = cv2.blur(grayscale_2,(k, k))


  # Morphology operations
  img_erosion = cv2.erode(nlm, kernel, iterations=3)
  img_dilation = cv2.dilate(img_erosion, kernel, iterations=3)
  gradient = cv2.morphologyEx(img_dilation, cv2.MORPH_GRADIENT, kernel)

  # changing output dimension to (224,224,1) for the input of deep network
  output = np.asarray(gradient).reshape(224,224,1)
  
  return output

# Applying preprocessing on test and train set

X_train_processed =[]
for i in X_train:
  X_train_processed.append(preprocessing(i))


X_test_processed = []
for i in X_test:
  X_test_processed.append(preprocessing(i))


# Converting X_train, X_test to numpy arrays for deep neural network
X_train_processed_np = np.array(X_train_processed)
X_test_processed_np = np.array (X_test_processed,)


# print(X_train_processed_np.shape, X_test_processed_np.shape)

# Importing required libraries for creating the network

from tensorflow.keras.layers import Input, Conv2D, ZeroPadding2D, Dense, Flatten, Layer, Conv2DTranspose, Activation, Subtract, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D, Concatenate, GlobalAveragePooling2D, Add, MaxPool2D, concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import metrics
from keras.models import Sequential

# Defining the number of classes for the network
# we will have one output that is either 1 or 0: 1 means malignant and 0 means benign
num_classes = 1


### Model 

input_shape = (224,224,1)

model = Sequential ()

model. add(Conv2D(32, kernel_size=(3,3),activation='relu' ,padding = 'Same' , input_shape=input_shape))
model .add(Conv2D(32,kernel_size=(3,3), activation='relu',padding = 'same'))
model. add(MaxPool2D(pool_size=(2,2)))
model. add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu',padding ='Same'))
model. add(Conv2D(64, (3, 3), activation='relu',padding ='Same'))
model. add(MaxPool2D(pool_size = (2, 2)))
model. add(Dropout(0.40))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model. add(Dropout(0.5))
model. add(Dense(num_classes, activation='sigmoid'))


model.summary()

# Training the model

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=["accuracy"] )

history = model.fit(X_train_processed_np,y_train, shuffle=True, epochs=5, batch_size=32,validation_data=(X_test_processed_np,y_test))

# Prediction on test set
skin_cancer_predict = model.predict(X_test_processed_np)
y_pred = np.round(skin_cancer_predict).tolist()








