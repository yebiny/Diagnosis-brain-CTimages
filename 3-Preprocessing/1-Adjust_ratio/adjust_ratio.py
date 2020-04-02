import numpy as np
import random
import os, sys
import matplotlib.pyplot as plt

sys.path.append('../../')
from  help_printing import *

def adjust_ratio(id_np, label_np):

    idx_1 = np.where(label_np == 1)[0]
    idx_0 = np.where(label_np == 0)[0]

    print( '  # of Label 1: ', len(idx_1))
    print( '  # of Label 0: ', len(idx_0))

    print( '---> Sampling the same number.')
    idx_1 = idx_1.tolist()
    idx_0 = idx_0.tolist()
    idx_0 = random.sample(idx_0, len(idx_1))
    print( '---> Shuffling data order.')
    idx_list = idx_1 + idx_0
    np.random.shuffle(idx_list)

    id_out = []
    label_out = []
    for i in range(len(idx_list)):
        id_out.append(id_np[idx_list[i]])
        label_out.append(label_np[idx_list[i]])
        
    id_out = np.array(id_out, str)
    label_out = np.array(label_out, int)
    return id_out, label_out

def hist_type_array(label_data, type_name, save_dir):
	n, bins, patches = plt.hist(label_data, 
                               bins = 2, rwidth = 0.8, 
                               color = 'orange')
	plt.xlabel('Label')
	plt.xticks(np.arange(0, 1, step = 2))
	plt.ylabel('Counts')
	plt.title(type_name)
	plt.savefig(save_dir + '/hist.png', dpi = 300)
    

def main():
	data_dir = str(input("- Enter the directory containing DICOM images : "))
	print(print_types)
	data_type = input("- Enter the subtype number : ")
	data_type = types[int(data_type)-1]
			
	np_dir = '../../2-EDA/res_%s/%s/'%(data_dir,data_type)
	if_not_exit(np_dir)

	print('\n* Dataset: [ %s ]'%(np_dir))

	print( '---> Loading numpy data.')
	id_np = np.load(np_dir + '/id_data.npy')
	label_np = np.load(np_dir + '/label_data.npy')
	id_out, label_out = adjust_ratio(id_np, label_np)
		
	save_dir = 'res_' + data_dir
	if_not_make(save_dir)

	save_dir = save_dir + '/' + data_type
	if_not_make(save_dir)

	print('---> Saving numpy data.')
	np.save(save_dir + '/id_data', id_out)
	np.save(save_dir + '/label_data', label_out)
	print('---> Drawing [ %s ] type ratio histogram'%data_type)
	hist_type_array(label_out, data_type, save_dir)

	summary(save_dir, [id_out, label_out])	
if __name__=='__main__':
	main()
