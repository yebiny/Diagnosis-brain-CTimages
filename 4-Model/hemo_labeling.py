# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:21:25 2020

@author: MG
"""

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import cv2 as cv

base_dir = 'E:\\Dataset\\rsna-intracranial-hemorrhage-detection\\brain_diagnosis\\3-Preprocessing\\2-HU_Window\\res_stage_2_train'

categories = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
nb_classes = len(categories)

image_w = 128
image_h = 128
pixels = image_w * image_h * 3

# 이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, cat in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = base_dir + "\\" + cat
    files = np.load(image_dir + "\\img_data.npy")
    for i in files:
        data = cv.normalize(i, None, 0, 1, cv.NORM_MINMAX)
        X.append(data)
        Y.append(label)
        
X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, shuffle = True, stratify = Y, random_state = 34)
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