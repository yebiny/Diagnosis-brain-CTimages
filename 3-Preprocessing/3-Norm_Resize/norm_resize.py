import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os,sys
import seaborn as sns
import scipy.ndimage

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
        i += 1
        
    print('* Final image data shape: ', np.shape(norm_resize_imgs))    
    return norm_resize_imgs    

def main():
    data_dir = str(input("enter the data directory: "))#'dcm_test'#sys.argv[1]
    data_type= str(input("enter the subtype: "))#'any'#sys.argv[2]
    print('\n* Dataset from [ %s ] and desease type is [ %s ]'%(data_dir, data_type))
	
    np_dir = '../2-HU_Window/res_%s/%s/'%(data_dir,data_type)
	
    if not os.path.exists(np_dir):
        print("!Error! There is not %s  Please make pre-dataset."%(np_dir))
        sys.exit()

    print( '\n--- Loading numpy data from ', np_dir )
    id_np = np.load(np_dir + '/id_data.npy')
    #img_np = np.load(np_dir + '/img_data.npy')
    label_np = np.load(np_dir + '/label_data.npy')
    load_dir = np_dir + '/img_data'
    print('* length of id.npy :', len(id_np))
    #print('* length of img.npy :', len(img_np))
    print('* length of label.npy :', len(label_np))	
	
    save_dir = 'res_'+data_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_dir = save_dir + '/' + data_type
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/pngs/')
    
    
    print('* Save dir : ', save_dir)

    print('\n--- Transform images... Nomalize and Reshape')
    img_size = 200
    norm_resize_imgs = norm_resize_stream(id_np, load_dir, img_size, save_dir + '/pngs')
    print('\n--- png files are saved in', save_dir + '/pngs')

    print( '\n--- Saving numpy data... (img, id, label)')
    re_img_np = np.array(norm_resize_imgs)
    
    np.save(save_dir + '/img_data', re_img_np)
    np.save(save_dir + '/id_data', id_np)
    np.save(save_dir + '/label_data', label_np)


if __name__=='__main__':
	main()


#2번에서 png 파일 저장하면, 그거를 가지고 내가 불러와서 놈하고 리사이즈(128, 128)