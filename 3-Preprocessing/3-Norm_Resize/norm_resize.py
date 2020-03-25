import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os,sys
import seaborn as sns
import scipy.ndimage

def norm_resize_stream(img_np, id_np, img_size, save_dir):
    
    norm_resize_imgs=[]
    for i in range(len(img_np)):
        img = img_np[i]
        img = cv.normalize(img.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX)
        img = cv.resize(img, dsize=(img_size, img_size), interpolation=cv.INTER_LINEAR)
        cv.imwrite(save_dir+'/pngs/'+id_np[i]+'.png', img)
        norm_resize_imgs.append(img)
        
    print('* Final image data shape: ', np.shape(norm_resize_imgs))    
    return norm_resize_imgs    

def main():
	data_dir = sys.argv[1]
	data_type= sys.argv[2]
	print('* Dataset from [ %s ] and desease type is [ %s ]'%(data_dir, data_type))
	
	np_dir = '../2-HU_Window/res_%s/%s/'%(data_dir,data_type)
	
	if not os.path.exists(np_dir):
		print("!Error! There is not %s  Please make pre-dataset."%(np_dir))
		sys.exit()

	print( '--- Loading numpy data from ', np_dir )
	id_np = np.load(np_dir+'/id_data.npy')
	img_np = np.load(np_dir+'/img_data.npy')
	label_np = np.load(np_dir+'/label_data.npy')
	print('* length of id.npy :', len(id_np))
	print('* length of img.npy :', len(img_np))
	print('* length of label.npy :', len(label_np))	
	
	save_dir = 'res_'+data_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	save_dir = save_dir+'/'+data_type
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
		os.makedirs(save_dir+'/pngs/')
    
	print('* Save dir : ', save_dir)

	print('--- Transform images... Nomalize and Reshape')
	img_size = 128
	norm_resize_imgs = norm_resize_stream(img_np, id_np, img_size, save_dir)
	print('--- png files are saved in', save_dir+'/pngs')

	print( '--- Saving numpy data... (img, id, label)')
	re_img_np = np.array(norm_resize_imgs)
	np.save(save_dir+'/img_data',re_img_np)
	np.save(save_dir+'/id_data', id_np)
	np.save(save_dir+'/label_data', label_np)


if __name__=='__main__':
	main()
