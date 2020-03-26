# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:30:36 2020

@author: MG
"""

import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os
import seaborn as sns
import scipy.ndimage

from keras import layers
from keras.applications import DenseNet121
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm

#%%
base_path = '../1-Dataset/'

train = pd.read_csv(base_path + '/stage_pre_train.csv')

train.head(12)

train.shape   #(4045572, 2)

newtable = train.copy()
train.Label.isnull().sum()   #---------->NaN의 갯수 합
#isnull : 관측치가 결측이면 True, 결측이 아니면 False의 boolean 값

#%%
#image example

train_images_path = base_path + '\\stage_pre_train\\'
train_images_list = [s for s in listdir(train_images_path) if isfile(join(train_images_path, s))]
test_images_path = base_path + '\\stage_pre_test\\'
test_images_list = [s for s in listdir(test_images_path) if isfile(join(test_images_path, s))]

print('5 Training images list', train_images_list[ : 5])
 
#%%
print('Total File sizes')
for f in os.listdir(base_path):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize(base_path + '\\' + f) / 1000000, 2)) + 'MB')
        #font size 30, 왼쪽정렬

#%%
print('Number of train images:', len(train_images_list))
print('Number of test images:', len(test_images_list))

#%%
#checking images

fig=plt.figure(figsize=(15, 10))
columns = 5; rows = 1
for i in range(1, columns * rows + 1):
    dcm = pydicom.dcmread(train_images_path + train_images_list[i - 1])
    fig.add_subplot(rows, columns, i)
    plt.imshow(dcm.pixel_array, cmap=plt.cm.bone)
    fig.add_subplot

print(dcm)

im_pixel = dcm.pixel_array
print(type(im_pixel))
print(im_pixel.dtype)
print(im_pixel.shape)




#%%
#plotting HU
def get_img_hu(patient_idx):
    dcm = pydicom.dcmread(train_images_path + train_images_list[patient_idx])
    image = dcm.pixel_array
    image = image.astype(np.int16)
    
    if np.min(image) < -1024:
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image < -1024] = 0
    
        # Convert to Hounsfield units (HU)
        intercept = dcm.RescaleIntercept
        slope = dcm.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
            
        image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


one_patient_HU = get_img_hu(4)

plt.hist(one_patient_HU.flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


one_patient_gray = cv.normalize(one_patient_HU.astype(np.float64), None, 0, 255, cv.NORM_MINMAX)



patient_hu = []
idx = 0

for idx in range(len(train_images_list)):
    tmp = get_img_hu(idx)
    patient_hu.append(tmp)

patient_gray = []
idx = 0

for idx in range(len(train_images_list)):
    tmp = cv.normalize(get_img_hu(idx).astype(np.float64), None, 0, 255, cv.NORM_MINMAX)
    patient_gray.append(tmp)

patient_hu = np.array(patient_hu)
patient_gray = np.array(patient_gray)


#%%
#Visualization of data
sns.countplot(train.Label) #csv 파일 0, 1 비율
train.Label.value_counts()

pylab.imshow(patient_hu[0], cmap = pylab.cm.gist_gray)
pylab.axis('on')



#%%
#Working newTable

train['Sub_type'] = train['ID'].str.split("_", n = 3, expand = True)[2]
train['PatientID'] = train['ID'].str.split("_", n = 3, expand = True)[1]

train.head()

gbSub = train.groupby('Sub_type').sum()
gbSub

sns.barplot(y = gbSub.index, x = gbSub.Label, palette = "deep")

fig = plt.figure(figsize = (10, 8))
sns.countplot(x = "Sub_type", hue = "Label", data = train)
plt.title("Total Images by Subtype")

#%%
def window_image(img, window_center,window_width, intercept, slope):

    img = (img * slope +intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img 

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

train_images_path

def view_images(images, title = '', aug = None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        
        data = pydicom.read_file(os.path.join(train_images_path, 'ID_' + images[im] + '.dcm'))
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)


        i = im // width
        j = im % width
        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
        
    plt.suptitle(title)
    plt.show()

view_images(train[(train['Sub_type'] == 'any') & (train['Label'] == 1)][:10].PatientID.values, title = 'Images of hemorrhage epidural')
view_images(train[(train['Sub_type'] == 'any') & (train['Label'] == 0)][:10].PatientID.values, title = 'Images of hemorrhage epidural')








