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

def npy_maker(id_np, load_dir):
    filelist = [f for f in glob.glob(load_dir + '/*.png', recursive = True)]
    #clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(9, 9))
    norm_imgs = []
    i = 0
    for file in filelist:
    #for i in range(len(img_np)):
        #img = img_np[i]
        img = cv.imread(file, cv.IMREAD_COLOR)
        img = cv.resize(img, dsize = (128, 128), interpolation = cv.INTER_AREA)
        #img_norm = cv.normalize(img, None, 0, 1, cv.NORM_MINMAX)
        img_norm = img / 255.0
        
        norm_imgs.append(img_norm)
        i += 1
    
    return norm_imgs

def main():
		
	data_dir = str(input("- Enter the directory containing DICOM images : "))
	print(print_types)
	data_type = input("- Enter the subtype number : ")
	data_type = types[int(data_type)-1]
	
	np_dir = '../3-Norm_Resize/res_%s/%s/'%(data_dir,data_type)
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

	#save_img_dir = save_dir+'/pngs'	
	#if_not_make(save_img_dir)
	
	
	npy_imgs = npy_maker(id_np, img_dir)
	print('---> Saving numpy data.')
	re_img_np = np.array(npy_imgs)
	
	#np.save(save_dir + '/id_data', id_np)
	#np.save(save_dir + '/label_data', label_np)
	np.save(save_dir + '/img_data_color', re_img_np)
	
	summary(save_dir, [id_np, label_np, re_img_np])
if __name__=='__main__':
	main()
