import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os,sys
import seaborn as sns
import scipy.ndimage

def show_img(img, figsize=(4,4)):
    fig=plt.figure(figsize=figsize)
    plt.imshow(img, cmap=plt.cm.bone)
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
    print('* # of target images: %s'%(len(id_np)))
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
	data_dir = sys.argv[1]
	data_type= sys.argv[2]
	print('* Dataset from [ %s ] and desease type is [ %s ]'%(data_dir, data_type))
	
	img_dir = '../../1-Dataset/'+data_dir
	np_dir = '../1-Adjust_ratio/res_%s/%s/'%(data_dir,data_type)
	
	if not os.path.exists(np_dir):
		print("!Error! There is not %s  Please make npy dataset at 2-EDA first."%(np_dir))
		sys.exit()
	
	print( '--- Loading numpy data from ', np_dir )
	id_np = np.load(np_dir+'/id_data.npy')
	label_np = np.load(np_dir+'/label_data.npy')
	
	print('--- Transform images... H.U and Windowing')
	hu_window_imgs = hu_window_stream(img_dir, id_np)

	print('* Output image dataset shape is :', np.shape(hu_window_imgs))

	save_dir = 'res_'+data_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	save_dir = save_dir+'/'+data_type
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	    
	print( '--- Saving numpy data...')
	print( '* save dir : ', save_dir)

	img_np = np.array(hu_window_imgs)
	np.save(save_dir+'/img_data',img_np)
	np.save(save_dir+'/id_data', id_np)
	np.save(save_dir+'/label_data', label_np)

if __name__=='__main__':
	main()
