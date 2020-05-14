# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:40:12 2020

@author: MIT-DGMIF
"""


from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

base_dir = 'E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\3-Preprocessing\\2-HU_Window\\res_stage_2_train\\all'

img_data = np.load(base_dir + '\\img_data.npy', allow_pickle=True)
label_data = np.load(base_dir + '\\label_data.npy', allow_pickle=True)
        

X = np.array(img_data)

Y = []
error = []
for i in range(len(label_data)):
    t = np.array(label_data[i])
    Y.append(t)
    if len(t) != 6:
        error.append(i)
    print(i)

Y = np.array(Y)
Y = np.uint8(Y)





# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
xy = (X_train, X_test, y_train, y_test)

print('>>> data 저장중 ...')

base_path = 'E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\4-Model'

dataset_path = os.path.join(base_path + '\\dataset')
if not(os.path.exists(dataset_path)):
    os.mkdir(dataset_path)
    
np.save(dataset_path + '\\X_train.npy', X_train)
np.save(dataset_path + '\\y_train.npy', y_train)
np.save(dataset_path + '\\X_test.npy', X_test)
np.save(dataset_path + '\\y_test.npy', y_test)
print("ok,", len(Y))