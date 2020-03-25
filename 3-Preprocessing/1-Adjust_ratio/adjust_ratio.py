import numpy as np
import random
import os, sys

def adjust_ratio(id_np, label_np):

    idx_1 = np.where(label_np==1)[0]
    idx_0 = np.where(label_np==0)[0]

    print( '* # of Label 1: ', len(idx_1))
    print( '* # of Label 0: ', len(idx_0))

    print( '--- Sampling the same number...')
    idx_1 = idx_1.tolist()
    idx_0 = idx_0.tolist()
    idx_0 = random.sample(idx_0, len(idx_1))
    print( '--- Shuffling data order ...')
    idx_list = idx_1+idx_0
    np.random.shuffle(idx_list)
    print( '* Final data length : ', len(idx_list))

    id_out = []
    label_out = []
    for i in range(len(idx_list)):
        id_out.append(id_np[idx_list[i]])
        label_out.append(label_np[idx_list[i]])
        
    id_out = np.array(id_out, str)
    label_out = np.array(label_out, int
                        )
    return id_out, label_out

def main():
	data_dir = sys.argv[1]
	data_type= sys.argv[2]
	print('* Dataset from [ %s ] and desease type is [ %s ]'%(data_dir, data_type))
	
	np_dir = '../../2-EDA/res_%s/%s/'%(data_dir,data_type)
	if not os.path.exists(np_dir):
		print("!Error! There is not %s  Please make npy dataset at 2-EDA first."%(np_dir))
		sys.exit()
	
	print( '--- SLoading numpy data...')
	id_np = np.load(np_dir+'/id_data.npy')
	label_np = np.load(np_dir+'/label_data.npy')
	
	id_out, label_out = adjust_ratio(id_np, label_np)
	
	save_dir = 'res_'+data_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	
	save_dir = save_dir+'/'+data_type
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	    
	print( '--- Saving numpy data...')
	print( '* save dir : ', save_dir)
	np.save(save_dir+'/id_data', id_out)
	np.save(save_dir+'/label_data', label_out)

if __name__=='__main__':
	main()
