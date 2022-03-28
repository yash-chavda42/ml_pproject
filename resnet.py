import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import glob
from skimage import io
#import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

import keras
import keras.layers as layers
#from keras import regularizers
#from keras.layers import Dropout

import tensorflow as tf
from tensorflow.keras.models import Sequential,Model ##squnce of process

import tensorflow as tf
from tensorflow.keras.models import Sequential,Model ##squnce of process
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout,Conv2D,MaxPooling2D   #bipertate graph 
from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras.utils import to_categorical #for catagorical data


class0_path=r"E:\sem6\ml\dataset_project\augmented_image\0\*.jpg"
class1_path=r"E:\sem6\ml\dataset_project\augmented_image\1\*.jpg"
class2_path=r"E:\sem6\ml\dataset_project\augmented_image\2\*.jpg"
class3_path=r"E:\sem6\ml\dataset_project\augmented_image\3\*.jpg"



image = list()
classes = list()

for file in glob.glob(class0_path):     
  img=cv2.imread(file)
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(0)#append all the class in classes


for file in glob.glob(class1_path):     
  img=cv2.imread(file)
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(1)#append all the class in classes


for file in glob.glob(class2_path):     
  img=cv2.imread(file)
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(2)#append all the class in classes


for file in glob.glob(class3_path):     
  img=cv2.imread(file)
  img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  image.append(img2)#append all the image in imge
  classes.append(3)#append all the class in classes



print(image[0])
print(len(image))
print(len(classes))
print(classes)

x_train, x_val, y_train, y_val=0,0,0,0
#split train and test
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(image, classes, test_size=0.2, random_state=4)

x_train=np.array(x_train)
y_train=np.array(y_train)
x_val=np.array(x_val)
y_val=np.array(y_val)


#reshape image
x_train=x_train.reshape(-1,256,1600,1)
x_val=x_val.reshape(-1,256,1600,1)

print(x_train.shape)


#unique classes
classes=np.unique(y_train)
nclasses=len(classes)
print(classes)
print(nclasses)

#this is for transfar learning
pretrained_model=tf.keras.applications.ResNet50(
    include_top=False,#false means i am going to use my own i/o an o/p layer becasue image dim is differnent
    input_tensor=None,
    input_shape=None,
    pooling='avg',
    classes=4,
    
)

for layer in pretrained_model.layers:
        layer.trainable=False

#resnet model
resnet_model = Sequential()
resnet_model.add(pretrained_model)

resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu'))#512 nueurons
resnet_model.add(Dense(4, activation='softmax'))#4 classes


resnet_model.summary()

resnet_model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

#history = resnet_model.fit(train_ds, validation_data=val_ds, epochs=10)
model_train=resnet_model.fit(x_train,y_train,batch_size=20,epochs=10,verbose=1,validation_data=(x_val,y_val))#verbose is show process



