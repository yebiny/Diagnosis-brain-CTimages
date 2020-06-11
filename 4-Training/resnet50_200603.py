# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 13:41:11 2020

@author: MIT-DGMIF
"""

from keras import utils, models, layers, optimizers
from keras.models import Model, load_model, Sequential

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Add
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16, VGG19, ResNet50

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras import backend as K
from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import keras

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import os, sys

import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model


import numpy as np
import os
from PIL import Image
import os, glob
import matplotlib.pyplot as plt

base_path = 'E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\4-Model'
dataset_path = os.path.join(base_path + '\\dataset')

X_train = np.load(dataset_path + '\\X_train.npy', allow_pickle=True)
y_train = np.load(dataset_path + '\\y_train.npy', allow_pickle=True)
X_test = np.load(dataset_path + '\\X_test.npy', allow_pickle=True)
y_test = np.load(dataset_path + '\\y_test.npy', allow_pickle=True)


inputs = Input(shape = (256, 256, 3))

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(16, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1))(x) 
    return x   
  
def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)      
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(16, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(16, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(16, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
  
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(32, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(32, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
  
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(64, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
  
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x


from keras import backend as K
import tensorflow as tf

# Compatible with tensorflow backend

def focal_loss(gamma=2., alpha=.25):

	def focal_loss_fixed(y_true, y_pred):

		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))

		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

	return focal_loss_fixed


#%% 
    '''
x = conv1_layer(inputs)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
#x = Dropout(0.25)(x)
x = conv5_layer(x)
#x = Dropout(0.25)(x) 
x = GlobalAveragePooling2D()(x)
#x = Dense(1024)(x)
x = BatchNormalization()(x)
#x = Dense(2048)(x)
#x = Activation('relu')(x)
x = Dense(1024)(x)
x = Activation('relu')(x)
#x = Dense(64)(x)
#x = Activation('relu')(x)
#x = Dense(64)(x)
#x = Activation('relu')(x)
x = Dropout(0.3)(x)
#x = BatchNormalization()(x)
#x = Activation('relu')(x)
outputs = Dense(5, activation='sigmoid')(x)
model = Model(inputs, outputs)
#model = multi_gpu_model(resnet50, gpus = None)
model.summary()
model.compile(optimizer=optimizers.Adam(lr = 1e-6), loss = [focal_loss(gamma = 2, alpha = .25)],  metrics=['binary_accuracy'])
#model.compile(optimizer=optimizers.Adam(lr = 1e-5), loss = focal_loss,  metrics=['binary_accuracy'])
'''

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
x = base_model.output
#x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
output = Dense(5, activation = 'sigmoid')(x)
model = Model(base_model.input, output)
#model = multi_gpu_model(resnet50, gpus = None)
model.summary()
model.compile(optimizer=optimizers.Adam(lr = 1e-4), loss = [focal_loss(gamma = 2, alpha = .25)],  metrics=['binary_accuracy'])

hdf5_file = "E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\5-Results\\weight_Classification_200611_Adam.hdf5"

#datagenerator
batch_size = 16
epochs = 10
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_data_flow = train_datagen.flow(X_train, y_train, batch_size = batch_size)
val_data_flow = val_datagen.flow(X_test, y_test, batch_size = batch_size)


early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\5-Results\\best_model_classification_200611.h5', monitor='val_loss', mode='min', save_best_only=True)

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    #early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

    #history = model.fit_generator(train_data_flow, epochs = epochs, steps_per_epoch = 1500, verbose = 2, validation_data = val_data_flow, validation_steps = 125, workers = 4, callbacks=[ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=30, verbose=1, mode='auto', min_lr=1e-10), mc])
    #callbacks = [checkpoint])    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose = 1, validation_split = 0.2, callbacks=[early_stopping, mc, ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience = 5, verbose=1, mode='auto', min_lr=1e-8)])
    model.save_weights(hdf5_file)
    model.save('E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\5-Results\\model_Classification_200611_Adam.h5')
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].set_title('loss')
    ax[0, 0].plot(history.history['loss'], 'r')
    ax[0, 1].set_title('acc')
    ax[0, 1].plot(history.history['binary_accuracy'], 'b')

    ax[1, 0].set_title('val_loss')
    ax[1, 0].plot(history.history['val_loss'], 'r--')
    ax[1, 1].set_title('val_acc')
    ax[1, 1].plot(history.history['val_binary_accuracy'], 'b--')
    



loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# 7. 모델 사용하기
xhat = X_test[0:100]
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)  
    
    


from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import cv2 as cv

# 분류 대상 카테고리 선택하기 

categories = ['epidural','intraparenchymal','intraventricular', 'subarachnoid', 'subdural', 'any']
nb_classes = len(categories)
class_names =['negative', 'positive']
# 이미지 크기 지정 
image_w = 128
image_h = 128
pixels = image_w * image_h * 3

predictions = model.predict(X_test)
pred =  predictions.astype("float16")

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    norm_img = cv.normalize(img.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX)
    plt.imshow(norm_img, cmap=plt.cm.bone)
    
    predicted_label = []
    for idx in range(len(predictions_array)):
        predicted_label.append(round(predictions_array[idx]))
    predicted_label = np.array(predicted_label).astype(np.uint8)
    
    if list(predicted_label) == list(true_label):
        color = 'blue'
    else:
        color = 'red'
    pred_bool = predicted_label == 1
    label_bool = true_label == 1    
    categor = np.array(categories)
    plt.xlabel("{} {}% ({})".format(str(categor[pred_bool]), str(100 * (predictions_array)), str(categor[label_bool])), color = color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(nb_classes), predictions_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = []
    for idx in range(len(predictions_array)):
        predicted_label.append(round(predictions_array[idx]))
    predicted_label = np.array(predicted_label).astype(np.uint8)
    
    thisplot[predicted_label[0]].set_color('red')
    thisplot[true_label[0]].set_color('blue')
    
num_rows = 4
num_cols = 4
start_point = 10
num_images = num_rows * num_cols
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i + start_point, predictions, y_test, X_test)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i + start_point, predictions, y_test)
plt.show()


    
    
    
    