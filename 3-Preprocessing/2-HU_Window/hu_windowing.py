import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os,sys
import seaborn as sns
import scipy.ndimage

sys.path.append('../../')
from  help_printing import *

def show_img(img, figsize = (4, 4)):
    fig = plt.figure(figsize = figsize)
    plt.imshow(img, cmap = plt.cm.bone)
    plt.show()


def get_hu_img(img, slope, intercept):
    img = img.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    if np.min(img) < -1024:
        img[img < -1024] = 0
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
    window_img[hu_img < img_min] = img_min
    window_img[hu_img > img_max] = img_max

    return window_img

def hu_window_stream(img_dir, id_np):
    hu_window_imgs = []
    for i in range(len(id_np)):
        dcm = pydicom.dcmread('%s/ID_%s.dcm'%(img_dir, id_np[i]))
        img = dcm.pixel_array

        
        slope = dcm.RescaleSlope 
        intercept = dcm.RescaleIntercept
        hu_img = get_hu_img(img,slope, intercept)

        center = dcm.WindowCenter
        width  = dcm.WindowWidth
        window_img = get_window_img(hu_img, center, width)
        
        hu_window_imgs.append(window_img)
    return hu_window_imgs

def main():
	data_dir = str(input("- Enter the directory containing DICOM images : "))
	print(print_types)
	data_type = input("- Enter the subtype number : ")
	data_type = types[int(data_type)-1]
	
	img_dir = '../../1-Dataset/' + data_dir
	if_not_exit(img_dir)
	np_dir = '../1-Adjust_ratio/res_%s/%s/'%(data_dir,data_type)
	if_not_exit(np_dir)
	
	print('\n* Dataset: [ %s ] and [ %s ]'%(np_dir, img_dir))	
	
	print('---> Loading numpy data ')
	id_np = np.load(np_dir + '/id_data.npy')
	label_np = np.load(np_dir + '/label_data.npy')
	
	print('---> Transform images... H.U and Windowing')
	hu_window_imgs = hu_window_stream(img_dir, id_np)
	
	save_dir = 'res_' + data_dir
	if_not_make(save_dir)
	
	save_dir = save_dir + '/' + data_type
	if_not_make(save_dir)
	
	save_img_dir = save_dir + '/pngs'
	if_not_make(save_img_dir)

	print( '---> Saving png files.')
	img_np = np.array(hu_window_imgs)
	for i in range(len(img_np)):
		cv.imwrite(save_img_dir + '/' + str(id_np[i]) + '.png', img_np[i])
    
	#np.save(save_dir+'/img_data',img_np)
	print( '---> Saving numpy data.')
	np.save(save_dir + '/id_data', id_np)
	np.save(save_dir + '/label_data', label_np)
	
	summary(save_dir, [id_np, label_np, img_np])
if __name__=='__main__':
	main()
