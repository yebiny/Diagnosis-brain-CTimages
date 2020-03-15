# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:59:10 2020

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
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from keras.preprocessing import image
#%%
#load the dicom image list
base_path = 'C:\\Users\\MG\\Desktop\\brain_diagnosis'
train_images_path = img_path + '\\1-Dataset\\stage_pre_train\\'
train_images_list = [s for s in listdir(train_images_path) if isfile(join(train_images_path, s))]
#test_images_path = base_path + '\\stage_pre_test\\'
#test_images_list = [s for s in listdir(test_images_path) if isfile(join(test_images_path, s))]

#%%
#convert to HU images
save_hu_dir = base_path + '\\3-Preprocessing\\hu'
idx = 0
img_tmp = []

if not(os.path.exists(save_hu_dir)):
    os.mkdir(save_hu_dir)
    
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

for idx in range(len(train_images_list)):
    data = pydicom.dcmread(train_images_path + train_images_list[idx])
    image = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    image_windowed = window_image(image, window_center, window_width, intercept, slope)
    cv.imwrite(save_hu_dir + '\\' + str(train_images_list[idx]) + '.png', image_windowed)
    img_tmp.append(image_windowed)

train_x_imgs = np.array(img_tmp)    

plt.hist(train_x_imgs[1].flatten(), bins=80, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()

#%%
#resizing & normalization

imgdir = save_hu_dir
# if you want file of a specific extension (.png):
filelist = [f for f in glob.glob(imgdir + "**/*.png", recursive=True)]
save_res_dir = base_path + '\\3-Preprocessing\\resized(100)'
if not(os.path.exists(save_res_dir)):
    os.mkdir(save_res_dir)

frame_num = 1
img_size = 200
for file in filelist:
    img = cv.imread(file, cv.IMREAD_GRAYSCALE)
    norm_img = cv.normalize(img.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX)
    resize_image = cv.resize(norm_img, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
    name = os.path.basename(file)
    name = os.path.splitext(name)[0]
    cv.imwrite(save_res_dir + '\\' + name + '.png', resize_image)
    
    print(frame_num)
    frame_num += 1

#%%    
#load the label data
label_path = 'C:\\Users\\MG\\Desktop\\brain_diagnosis\\1-Dataset'
train = pd.read_csv(label_path + '\\stage_pre_train.csv', sep=",", dtype = 'unicode')
train['Sub_type'] = train['ID'].str.split("_", n = 3, expand = True)[2]
train['PatientID'] = train['ID'].str.split("_", n = 3, expand = True)[1]
train.head()
train.columns

#showing the subtype
group_sub = train.groupby('Sub_type').sum()
group_sub
sns.barplot(y = group_sub.index, x = group_sub.Label, palette = "deep")

#showing the subtype with label
fig = plt.figure(figsize = (10, 8))
sns.countplot(x = "Sub_type", hue = "Label", data = train)
plt.title("Total Images by Subtype")

#Load imgs according to label
image2train = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img(save_res_dir + '\\ID_' + train['PatientID'][i] + '.dcm.png', target_size = (100, 100))
    img = image.img_to_array(img)
    img = img / 255
    image2train.append(img)

X = np.array(image2train)
X.shape
plt.imshow(X[2])
train['Label'][2]

y = np.array(train.drop(['Label'], axis = 1))
y.shape

#split the dataset to training format
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.1)

dataset_path = os.path.join(base_path + '\\1-Dataset\\trainging_data')
if not(os.path.exists(dataset_path)):
    os.mkdir(dataset_path)
    
np.save(dataset_path + '\\X_train.npy', np.array(X_train))
np.save(dataset_path + '\\y_train.npy', np.array(y_train))
np.save(dataset_path + '\\X_val.npy', np.array(X_test))
np.save(dataset_path + '\\y_val.npy', np.array(y_test))

print(X_train[-1].shape, y_train[-1].shape)
print(X_test[-1].shape, y_test[-1].shape)









img_base_path = "D:\\H&E_dataset"
x_img_path = os.path.join(img_base_path + '\\H_V(40)')
y_img_path = os.path.join(img_base_path + '\\H_V(100)')

x_img = [cv.imread(x_img_path + '\\' + s, cv.IMREAD_GRAYSCALE) for s in os.listdir(x_img_path)]
y_img = [cv.imread(y_img_path + '\\' + s, cv.IMREAD_GRAYSCALE) for s in os.listdir(y_img_path)]


x_data = []
y_data = []


for indx in range(len(x_img)):
    norm_x = cv.normalize(x_img[indx].astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
    x_data.append(norm_x)
    norm_y = cv.normalize(y_img[indx].astype(np.float64), None, 0, 1, cv.NORM_MINMAX)
    y_data.append(norm_y)
    print(indx)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)
    
dataset_path = os.path.join(img_base_path, 'dataset')
if not(os.path.exists(dataset_path)):
    os.mkdir(dataset_path)
    
np.save('D:\\H&E_dataset\\dataset\\' + 'x_train_hv(40)_220025.npy', np.array(x_train))
np.save('D:\\H&E_dataset\\dataset\\' + 'y_train_hv(100)_220025.npy', np.array(y_train))
np.save('D:\\H&E_dataset\\dataset\\' + 'x_val_hv(40)_220025.npy', np.array(x_val))
np.save('D:\\H&E_dataset\\dataset\\' + 'y_val_hv(100)_220025.npy', np.array(y_val))

print(x_train[-1].shape, y_train[-1].shape)
print(x_val[-1].shape, y_val[-1].shape)













