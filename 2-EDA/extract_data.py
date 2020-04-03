import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from os import listdir, walk

sys.path.append('../')
from help_printing import *

def file_to_data(data_path):
    file_data = listdir(data_path)
    file_data = np.array(file_data)
    
    for i in range(len(file_data)):
        file_data[i] = file_data[i].split("_")[1].split(".")[0]
    return file_data

def make_type_array(data, file_data, type_name):
    data['subID'] = data['ID'].str.split("_", n = 3, expand = True)[1]
    data['subType'] = data['ID'].str.split("_", n = 3, expand = True)[2]
    type_filter = data['subType'] == type_name
    filter_data = np.array(data[type_filter])    
    #print(filter_data.shape, filter_data)
    
    id_data = []
    label_data = []
    
    for i in range(len(filter_data)):
        #if i%10000 == 0: print(i)
        if filter_data[i][2] in file_data:
            id_data.append(filter_data[i][2])
            label_data.append(filter_data[i][1])
        else:
            pass
    id_data = np.array(id_data, dtype = str)
    label_data = np.array(label_data, dtype = int)

    return id_data, label_data

def hist_type_array(label_data, type_name, save_dir):
	n, bins, patches = plt.hist(label_data, 
                               bins = 2, rwidth = 0.8, 
                               color = 'skyblue', 
                               )
	plt.xlabel('Label')
	plt.xticks(np.arange(0, 1, step = 2))
	plt.ylabel('Density')
	plt.title(type_name)
	plt.savefig(save_dir + '/hist.png', dpi = 300)
	#plt.show()
		

def main():
    csv_file = '../1-Dataset/stage_2_train.csv'
    #csv_file = '../1-Dataset/dcm_test.csv'
    if_not_exit(csv_file)
    csv_data = pd.read_csv(csv_file,sep = ",", dtype = 'unicode')
	
    dcm_dir = str(input("- Enter the directory containig DICOM images : "))
    dcm_path = '../1-Dataset/%s/'%(dcm_dir)
    if_not_exit(dcm_path)

    dcm_data = file_to_data(dcm_path)
    save_dir = 'res_' + dcm_dir
    if_not_make(save_dir)
	
    print(print_types)	
    type_name = input("- Enter the subtype number:")
    type_name = types[int(type_name)-1]

    save_dir = save_dir + '/' + type_name
    if_not_make(save_dir)

    print('\n* Dicom dir: [ %s ], Desease type : [ %s ]'%(dcm_dir, type_name))
    print('---> Finding ID and Label index from [ %s ].'%(csv_file))	
    id_data, label_data = make_type_array(csv_data, dcm_data, type_name)
    
    print('---> Saving numpy ID data.')
    np.save(save_dir + '/id_data', id_data)
    	
    print('---> Saving numpy Label data.')
    np.save(save_dir + '/label_data', label_data )
    
    print('---> Drawing [ %s ] type ratio histogram.'%(type_name))	
    hist_type_array(label_data, type_name, save_dir)
    
    index, count = np.unique(label_data, return_counts = True)
    summary(save_dir, [id_data, label_data], count)	
    
if __name__=='__main__':
	main()
