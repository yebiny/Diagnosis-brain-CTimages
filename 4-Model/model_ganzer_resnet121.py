from keras import utils, models, layers, optimizers
from keras.models import Model, load_model, Sequential

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Add

from keras.applications import DenseNet121

 
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.applications.resnet50 import ResNet50

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import keras

import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import os, sys

import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model




data_dir = 'stage_2_train'
data_type = 'subdural'
np_dir = '../3-Preprocessing/3-Norm_Resize/res_%s/%s'%(data_dir,data_type)

img_np = np.load('E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\3-Preprocessing\\4-NPY\\res_gist_ncar_scaled\\subdural\\img_data_color.npy')
label_np = np.load('E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\3-Preprocessing\\3-Norm_Resize\\res_gist_ncar_scaled\\subdural\\label_data.npy')

print(img_np.shape, label_np.shape)

class_names = ['other problem', str(data_type)]

plt.figure(figsize = (10, 10))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img_np[i], cmap = plt.cm.bone)
    plt.xlabel(class_names[label_np[i]])
plt.show()

label_=[]
for i in range(len(label_np)):
    if label_np[i]==0: label_.append([1,0])
    else: label_.append([0,1])
label_=np.array(label_)
print(label_.shape)

x_train, y_train, x_label, y_label = train_test_split(img_np, label_, test_size=0.2, random_state=1)
print('* input data shape:', x_train.shape, x_label.shape)
print('* output data shape:', y_train.shape, y_label.shape)

train_images = x_train.astype('float32')
#train_images = train_images / 255
#train_images = train_images[:, :, :, np.newaxis]

val_images = y_train.astype('float32')
#val_images = val_images / 255
#val_images = val_images[:, :, :, np.newaxis]

num_classes = len(class_names)

'''
train_labels = utils.to_categorical(y_train, num_classes)
test_labels = utils.to_categorical(y_test, num_classes)
'''
#%% yebiny model
"""
x = Input(shape=(x_shape[1], x_shape[2], x_shape[3]), dtype='float32', name='x')
    
y = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(x)
y = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
y = MaxPooling2D((2,2))(y)
    
y = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
y = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
y = MaxPooling2D((2,2))(y)
    
y = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
y = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
y = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01))(y)
y = MaxPooling2D((2,2))(y)
    
y = Flatten()(y)
y = Dense(2048, kernel_initializer='he_normal')(y)
y = Dense(512, kernel_initializer='he_normal')(y)
y = Dense(64, kernel_initializer='he_normal')(y)
y = Dense(2, activation='sigmoid')(y)

"""

#%%
densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(256,256,3))

def build_model():
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(6, activation='sigmoid', 
#                            bias_initializer=Constant(value=-5.5)))
    model.add(layers.Dense(2, activation='sigmoid'))
    
    
    
    return model

model = build_model()
model.summary()
model = multi_gpu_model(model, gpus=None)
model.compile(
#         loss=focal_loss,
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )


from keras.utils import plot_model
plot_model(model, to_file = 'dense121.png')

#%%
hdf5_file = np_dir + '/' + str(data_type) + '.hdf5'

if os.path.exists(hdf5_file):
    # 기존에 학습된 모델 불러들이기
    model.load_weights(hdf5_file)
else:
    # 학습한 모델이 없으면 파일로 저장
    #early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)

    history = model.fit(train_images, x_label, validation_data=(val_images, y_label), epochs=50, batch_size = 8, callbacks=[
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-04)])
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=32, callbacks=[early_stopping])
    model.save_weights(hdf5_file)
    
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    ax[0, 0].set_title('loss')
    ax[0, 0].plot(history.history['loss'], 'r')
    ax[0, 1].set_title('acc')
    ax[0, 1].plot(history.history['acc'], 'b')

    ax[1, 0].set_title('val_loss')
    ax[1, 0].plot(history.history['val_loss'], 'r--')
    ax[1, 1].set_title('val_acc')
    ax[1, 1].plot(history.history['val_acc'], 'b--')