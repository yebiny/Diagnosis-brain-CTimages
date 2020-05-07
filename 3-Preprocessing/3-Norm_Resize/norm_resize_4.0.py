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

def norm_resize_stream(id_np, load_dir, img_size, save_dir):
    filelist_1 = [f for f in glob.glob(load_dir + '/ori/*.png', recursive = True)]
    filelist_2 = [f for f in glob.glob(load_dir + '/dural/*.png', recursive = True)]
    filelist_3 = [f for f in glob.glob(load_dir + '/brain/*.png', recursive = True)]
       
    ori = []
    for file_1 in filelist_1:    
        img_ori_1 = cv.imread(file_1, cv.IMREAD_GRAYSCALE)
        img_1 = cv.normalize(img_ori_1, None, 0, 255, cv.NORM_MINMAX)
        img_resize_1 = cv.resize(img_1, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        ori.append(img_resize_1)
        
    dural = []
    for file_2 in filelist_2:
        img_ori_2 = cv.imread(file_2, cv.IMREAD_GRAYSCALE)
        img_2 = cv.normalize(img_ori_2, None, 0, 255, cv.NORM_MINMAX)
        img_resize_2 = cv.resize(img_2, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        dural.append(img_resize_2)
        
    brain = []
    for file_3 in filelist_3:
        img_ori_3 = cv.imread(file_3, cv.IMREAD_GRAYSCALE)
        img_3 = cv.normalize(img_ori_3, None, 0, 255, cv.NORM_MINMAX)
        img_resize_3 = cv.resize(img_3, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        brain.append(img_resize_3)
    
    norm_resize_imgs = []    
    for i in range(len(ori)):
        temp_img = cv.merge((ori[i], dural[i], brain[i]))
        plt.imsave(save_dir + '/' + str(id_np[i]) + '.png', temp_img, format='PNG', cmap='gist_ncar')
        #cv.imwrite(save_dir + '/' + str(id_np[i]) + '.png', img_resize)
        norm_resize_imgs.append(temp_img)
    
    plt.figure(figsize = (20, 10))
    plt.subplot(1, 3, 1)
    plt.hist(norm_resize_imgs[0][:,:,0].flatten(), bins = 80, color = 'r')
    plt.subplot(1, 3, 2)
    plt.hist(norm_resize_imgs[0][:,:,1].flatten(), bins = 80, color = 'g')
    plt.subplot(1, 3, 3)
    plt.hist(norm_resize_imgs[0][:,:,2].flatten(), bins = 80, color = 'b')
    plt.savefig(save_dir + '/../hist.png', dpi = 300)
    
    return norm_resize_imgs

def main():
	img_size = 256
	
	data_dir = str(input("- Enter the directory containing DICOM images : "))
	print(print_types)
	data_type = input("- Enter the subtype number : ")
	data_type = types[int(data_type)-1]
	
	np_dir = '../2-HU_Window/res_%s/%s/'%(data_dir,data_type)
	if_not_exit(np_dir)
	
	print('\n* Dataset from  [ %s ]'%(np_dir))
	
	print( '---> Loading numpy data' )
	id_np = np.load(np_dir + '/id_data.npy')
	label_np = np.load(np_dir + '/label_data.npy')
	
	img_dir = np_dir + '/pngs'
	
	save_dir = 'res_'+data_dir
	if_not_make(save_dir)

	save_dir = save_dir + '/' + data_type
	if_not_make(save_dir)	

	save_img_dir = save_dir+'/pngs'	
	if_not_make(save_img_dir)
	
	print('---> Transform images... Nomalize and Reshape')
	norm_resize_imgs = norm_resize_stream(id_np, img_dir, img_size, save_img_dir)
	print(' reshape size : ', img_size)
	print('---> Saving numpy data.')
	re_img_np = np.array(norm_resize_imgs)
	
	np.save(save_dir + '/id_data', id_np)
	np.save(save_dir + '/label_data', label_np)
	np.save(save_dir + '/img_data', re_img_np)
	
	summary(save_dir, [id_np, label_np, re_img_np])
if __name__=='__main__':
	main()
