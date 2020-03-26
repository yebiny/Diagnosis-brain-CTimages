import glob, pylab, pandas as pd
import pydicom, numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv

import matplotlib.pylab as plt
import os
import seaborn as sns
import scipy.ndimage

data_path = '../1-Dataset/'

def see_dir_size(dir_path):
	print('Total File sizes in '+dir_path)
	for f in os.listdir(dir_path):
		if 'zip' not in f:
			print(f.ljust(30) + str(round(os.path.getsize(dir_path + '/' + f) / 1000000, 4)) + 'MB')

class EDA_helper():
	def __init__(self, data_dir):
		
		self.path = data_path+data_dir+'/'
		self.list = [s for s in listdir(self.path) if isfile(join(self.path, s))]
		self.size = len(self.list)
		
		self.csvf = pd.read_csv(data_path + data_dir+'.csv' )
		self.dcms = [pydicom.dcmread(self.path+self.list[i]) for i in range(len(self.list))]
		self.imgs = [self.dcms[i].pixel_array for i in range(self.size)]
		
		self.slope = [self.dcms[i].RescaleSlope for i in range(self.size)]
		self.intercept = [self.dcms[i].RescaleIntercept for i in range(self.size)]
		self.label_type = self.csvf['ID'].str.split("_", n = 3, expand = True)[2][0:6].values

	def show_imgs(self, imgs, figsize=(15,6)):
		fig=plt.figure(figsize=figsize)
		columns = 5
		for i in range(10):
			fig.add_subplot(2,columns,i+1)
			plt.imshow(imgs[i], cmap=plt.cm.bone)
		plt.show()
	
	def get_hu_imgs(self):
		hu_imgs = []
		for i in range(self.size):
			img = self.imgs[i].astype(np.int16)
		
			# Set outside-of-scan pixels to 0
			# The intercept is usually -1024, so air is approximately 0
			if np.min(img) < -1024:
				img[img < -1024] = 0
			# Convert to Hounsfield units (HU)
			hu_img = (img * self.slope[i] + self.intercept[i])
			hu_imgs.append(hu_img)
		return hu_imgs	
	
	def get_window_imgs(self):
		
		windowed_imgs = []
		hu_imgs = self.get_hu_imgs()
		for i in range(self.size):
			img = hu_imgs[i]
			w_center = self.dcms[i].WindowCenter
			w_width  = self.dcms[i].WindowWidth

			if type(w_center) == pydicom.multival.MultiValue:
				w_center = w_center[0]
			if type(w_width) == pydicom.multival.MultiValue:
				w_width = w_width[0]

			img_min = w_center - w_width // 2
			img_max = w_center + w_width // 2
			img[img < img_min] = img_min
			img[img > img_max] = img_max
			
			windowed_imgs.append(img)
		return windowed_imgs

	def show_hu_graph(self, idx):
		hu_imgs = self.get_hu_imgs()
		hu_img = hu_imgs[idx]
		hu_img = np.array(hu_img, dtype=np.int16)

		plt.hist(hu_img.flatten(), bins=80, color='c')
		plt.xlabel("Hounsfield Units (HU)")
		plt.ylabel("Frequency")
		plt.show()
	
	def get_sub_label(self, idx):
		idx = idx*6
		sub_label = self.csvf['Label'][idx:idx+6].values	
		sub_id = self.csvf['ID'].str.split("_", n = 2, expand = True)[1][idx:idx+6].values
		return sub_id, sub_label

	def show_sub_label(self, idx):
		sub_id, sub_label = self.get_sub_label(idx)
		for i in range(0, 6):
			if sub_label[i] == 1: 
				c = 'red'
			else: 
				c = 'black'
			plt.text(520,60+80*i, self.label_type[i], color=c, size=12)
		img = self.get_window_imgs()[idx]
		plt.title(sub_id[0])
		plt.imshow(img, cmap=plt.cm.bone)
		plt.show()

	def plot_label(self, figsize=(12,5)):
		self.csvf['Sub_type'] = self.csvf['ID'].str.split("_", n = 3, expand = True)[2]
		fig = plt.figure(figsize = figsize)
		sns.countplot(x = "Sub_type", hue = "Label", data = self.csvf)
		plt.title("Total Images by Subtype")

				
