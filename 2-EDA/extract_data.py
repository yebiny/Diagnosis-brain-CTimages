import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from os import listdir, walk

def file_to_data(data_dir):
    data_path = '../1-Dataset/'+data_dir+'/'
    file_data = listdir(data_path)
    file_data = np.array(file_data)
    
    for i in range(len(file_data)):
        file_data[i] = file_data[i].split("_")[1].split(".")[0]
    return file_data

def make_type_array(data, file_data, type_name):
    data['subID'] = data['ID'].str.split("_", n=3, expand=True)[1]
    data['subType'] = data['ID'].str.split("_", n=3, expand=True)[2]
    type_filter = data['subType']==type_name
    filter_data = np.array(data[type_filter])    
    #print(filter_data.shape, filter_data)
    
    id_data = []
    label_data=[]
#for i in range(len(filter_data)):
#        if filter_data[i][2] in file_data:            
#            id_data.append(filter_data[i][2])
#            label_data.append(filter_data[i][1])
#        else: continue
    for i in range(len(file_data)):
        idx = np.where(filter_data==file_data[i])
        idx = idx[0][0]
        id_data.append(filter_data[idx][2])
        label_data.append(filter_data[idx][1])
    id_data = np.array(id_data, dtype=str)
    label_data = np.array(label_data, dtype=int)

    return id_data, label_data

def hist_type_array(label_data, type_name, save_dir):
	n, bins, patches= plt.hist(label_data, 
                               bins=2, rwidth=0.8, 
                               color = 'orange', 
                               )
	plt.xlabel('Label')
	plt.xticks(np.arange(0, 1, step=2))
	plt.ylabel('Density')
	plt.title(type_name)
	plt.savefig(save_dir+'/hist.png',dpi=300)
	#plt.show()
		

def main():
	csv_file = '../1-Dataset/dcm.csv'
	csv_data = pd.read_csv(csv_file,sep=",", dtype='unicode')
	
	dcm_dir = sys.argv[1] #or set 'dcm_500'
	dcm_data = file_to_data(sys.argv[1])
	save_dir = 'res_'+dcm_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	type_name = sys.argv[2] #or set 'any'/'epidural'/'intraparenchymal'/'subdural'/...'
	save_dir = save_dir+'/'+type_name
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	id_data, label_data = make_type_array(csv_data, dcm_data, type_name)
	hist_type_array(label_data, type_name, save_dir)

	print('* Dicom dir:', dcm_dir, 'Desease type: ', type_name)
	print('* ID data shape : ', id_data.shape)
	print('--- Saving numpy ID data')
	np.save(save_dir+'/id_data', id_data)
		
	print('* Label data shape : ', label_data.shape) 
	print('--- Saving numpy Label data')
	np.save(save_dir+'/label_data', label_data )
	
	index, count = np.unique(label_data, return_counts=True)
	print('* Counting  0: ', count[0], '1: ', count[1])
	print('* Ratio is ', count[1]/count[0])

if __name__=='__main__':
	main()
