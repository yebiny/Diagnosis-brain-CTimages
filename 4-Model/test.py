import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import os, sys


data_dir = 'dcm_500'
data_type = 'subdural'
np_dir = '../3-Preprocessing/res_%s/%s/3-3'%(data_dir,data_type)

img_np = np.load(np_dir + '/img_data.npy')
label_np = np.load(np_dir + '/label_data.npy')

print(img_np.shape, label_np.shape)

label_=[]
for i in range(len(label_np)):
    if label_np[i]==0: label_.append([1,0])
    else: label_.append([0,1])
label_=np.array(label_)
print(label_.shape)
x_train, x_test, y_train, y_test = train_test_split(img_np, label_np, test_size=0.2, random_state=1)
print('* input data shape:', x_train.shape, x_test.shape)
print('* output data shape:', y_train.shape, y_test.shape)


x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

print(x_train.shape)
print(y_train.shape)

from tensorflow.keras import layers, models

def my_model():

  x = keras.Input(shape=(128,128,1))
  
  y = layers.Conv2D(8,(3,3), padding='same')(x)
  y = layers.Activation('relu')(y)
  y = layers.MaxPool2D(pool_size=(3,3))(y)
  
  y = layers.Conv2D(16,(3,3), padding='same')(y)
  y = layers.Activation('relu')(y)
  y = layers.MaxPool2D(pool_size=(3,3))(y)
  
  y = layers.Flatten()(y)
  
  y = layers.Dense(1024)(y)
  y = layers.BatchNormalization()(y)
  y = layers.Activation('relu')(y)

  y = layers.Dense(128)(y)
  y = layers.BatchNormalization()(y)
  y = layers.Activation('relu')(y)
  
  y = layers.Dense(32)(y)
  y = layers.BatchNormalization()(y)
  y = layers.Activation('relu')(y)
  
  y = layers.Dense(1)(y)
  y = layers.Activation('sigmoid')(y)
  
  return models.Model(inputs=x, outputs=y)

model = my_model()
model.summary()

model.compile('adam','binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=2)
