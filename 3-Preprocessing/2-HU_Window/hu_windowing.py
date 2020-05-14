# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:29:04 2020

@author: MG
"""

import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os,sys
import seaborn as sns
import scipy.ndimage
import imageio

sys.path.append('../../')
from  help_printing_mg import *

def show_img(img, figsize = (4, 4)):
    fig = plt.figure(figsize = figsize)
    plt.imshow(img, cmap = plt.cm.bone)
    plt.show()


def get_hu_img(img, slope, intercept):
    img = img.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    #if np.min(img) < -1024:
        #img[img < -1024] = 0
    # Convert to Hounsfield units (HU)
    hu_img = (img * slope + intercept)
    
    return hu_img

def get_window_img(hu_img, center, width):

    window_img = hu_img
    
    if type(center) == pydicom.multival.MultiValue:
        center = center[0]
    if type(width) == pydicom.multival.MultiValue:
        width = width[0]

    img_min = center - width // 2
    img_max = center + width // 2
    window_img[window_img < img_min] = img_min
    window_img[window_img > img_max] = img_max

    return window_img

def hu_window_stream(img_dir, id_np, save_ori_dir, save_dural_dir, save_brain_dir):
    img_size = 128
    imgset = []
    for i in range(len(id_np)):
        dcm = pydicom.dcmread('%s/ID_%s.dcm'%(img_dir, id_np[i]))
        img = dcm.pixel_array

        
        slope = dcm.RescaleSlope 
        intercept = dcm.RescaleIntercept
        hu_img_1 = get_hu_img(img, slope, intercept)
        hu_img_2 = get_hu_img(img, slope, intercept)
        hu_img_3 = get_hu_img(img, slope, intercept)
        
        center = dcm.WindowCenter
        width  = dcm.WindowWidth
        
        window_img_ori = get_window_img(hu_img_1, center, width)
        window_img_subdural = get_window_img(hu_img_2, 80.0, 200.0)
        window_img_brain = get_window_img(hu_img_3, 40.0, 80.0)
        
        img_1 = cv.normalize(window_img_ori, None, 0, 255, cv.NORM_MINMAX)
        img_resize_1 = cv.resize(img_1, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        
        img_2 = cv.normalize(window_img_subdural, None, 0, 255, cv.NORM_MINMAX)
        img_resize_2 = cv.resize(img_2, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        
        img_3 = cv.normalize(window_img_brain, None, 0, 255, cv.NORM_MINMAX)
        img_resize_3 = cv.resize(img_3, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        
        #temp_img = cv.merge((img_resize_1, img_resize_2, img_resize_3))
        
        #plt.imsave(save_ori_dir + '/' + str(id_np[i]) + '.png', window_img_ori, cmap = 'bone')
        #plt.imsave(save_dural_dir + '/' + str(id_np[i]) + '.png', window_img_subdural, cmap = 'bone')
        #plt.imsave(save_brain_dir + '/' + str(id_np[i]) + '.png', window_img_brain, cmap = 'bone')
        #plt.imsave(save_set_dir + '/' + str(id_np[i]) + '.png', temp_img, cmap = 'bone')
        
        
        imgset.append(cv.merge((img_resize_1, img_resize_2, img_resize_3)))
        print(i)
    
    return imgset#, hu_subdural, hu_brain, imgset


def main():
    data_dir = str(input("- Enter the directory containing DICOM images : "))
    print(print_types)
    data_type = input("- Enter the subtype number : ")
    data_type = types[int(data_type)-1]
    
    img_dir = '../../1-Dataset/' + data_dir
    if_not_exit(img_dir)
    
    if data_type == 'normal':
        np_dir = '../../2-EDA/res_%s/%s/'%(data_dir,data_type)
    elif data_type == 'all':
        np_dir = '../../2-EDA/res_%s/%s/'%(data_dir,data_type)
    else:
        np_dir = '../1-Adjust_ratio/res_%s/%s/'%(data_dir,data_type)
        
    if_not_exit(np_dir)
    
    print('\n* Dataset: [ %s ] and [ %s ]'%(np_dir, img_dir))
    print('---> Loading numpy data ')
    
    id_np = np.load(np_dir + '/id_data.npy', allow_pickle=True)
    label_np = np.load(np_dir + '/label_data.npy', allow_pickle=True)
    
    print('---> Transform images... H.U and Windowing')
  
  
    #imgset = cv.merge((ori, subdural, brain))
    
    
    save_dir = 'res_' + data_dir
    if_not_make(save_dir)
    
    save_dir = save_dir + '/' + data_type
    if_not_make(save_dir)
    
    save_ori_dir = save_dir + '/pngs/ori'
    if_not_make(save_ori_dir)
    
    save_dural_dir = save_dir + '/pngs/dural'
    if_not_make(save_dural_dir)
    
    save_brain_dir = save_dir + '/pngs/brain'
    if_not_make(save_brain_dir)
    
    #save_set_dir = save_dir + '/pngs/set'
    #if_not_make(save_set_dir)
    
    imgset_np = hu_window_stream(img_dir, id_np, save_ori_dir, save_dural_dir, save_brain_dir)
    
    print( '---> Saving png files.')
    
    #ori_np = np.array(ori)
    #subdural_np = np.array(subdural)
    #brain_np = np.array(brain)
    '''
    imgset_np = []
    for i in range(len(ori_np)):
        plt.imsave(save_ori_dir + '/' + str(id_np[i]) + '.png', ori_np[i], cmap = 'bone')
        plt.imsave(save_dural_dir + '/' + str(id_np[i]) + '.png', subdural_np[i], cmap = 'bone')
        plt.imsave(save_brain_dir + '/' + str(id_np[i]) + '.png', brain_np[i], cmap = 'bone')
        imgset_np.append(cv.merge((ori_np[i], subdural_np[i], brain_np[i])))
    '''    
    print( '---> Saving numpy data.')
    np.save(save_dir + '/id_data', id_np)
    np.save(save_dir + '/label_data', label_np)
    np.save(save_dir + '/img_data', imgset_np)
    summary(save_dir, [id_np, label_np, imgset_np])
    
if __name__=='__main__':
    main()