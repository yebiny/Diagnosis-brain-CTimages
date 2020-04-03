import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os,sys
import seaborn as sns
import scipy.ndimage

sys.path.append('../')
from  help_printing import *

def norm_resize_stream(id_np, load_dir, img_size, save_dir):
    filelist = [f for f in glob.glob(load_dir + '/*.png', recursive = True)]
    norm_resize_imgs = []
    i = 0
    for file in filelist:
    #for i in range(len(img_np)):
        #img = img_np[i]
        img = cv.imread(file, cv.IMREAD_GRAYSCALE)
        img = cv.normalize(img.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX)
        img = cv.resize(img, dsize = (img_size, img_size), interpolation = cv.INTER_LINEAR)
        cv.imwrite(save_dir + '/' + str(id_np[i]) + '.png', img)
        norm_resize_imgs.append(img)
        i=i+1
    return norm_resize_imgs    

def main():
	img_size = 200
	
	data_dir = str(input("- Enter the directory containig DICOM images : "))
	print(print_types)
	data_type = input("- Enter the subtype number : ")
	data_type = types[int(data_type)-1]
	
	np_dir = './res_%s/%s/3-2'%(data_dir,data_type)
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
	save_dir = save_dir + '/3-3'
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
