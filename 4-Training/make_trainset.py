import numpy as np
import os, sys
from os import listdir
import sklearn
from sklearn.model_selection import train_test_split
sys.path.append('../')
from help_printing import *

def make_dataset(img_np, label_np, save_dir, split):
    x_train, x_test, y_train, y_test = train_test_split(img_np, label_np, test_size=split[2]/sum(split), random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=split[1]/sum(split[:-1]), random_state=1)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2],1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2],1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)

    np.save(save_dir + '/x_train', x_train) 
    np.save(save_dir + '/y_train', y_train) 
    np.save(save_dir + '/x_val', x_val) 
    np.save(save_dir + '/y_val', y_val) 
    np.save(save_dir + '/x_test', x_test) 
    np.save(save_dir + '/y_test', y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_dataset(load_dir):
    x_train = np.load(load_dir + 'x_train.npy')
    y_train = np.load(load_dir + 'y_train.npy')
    x_val = np.load(load_dir + 'x_val.npy')
    y_val = np.load(load_dir + 'y_val.npy')
    x_test = np.load(load_dir + 'x_test.npy')
    y_test = np.load(load_dir + 'y_test.npy')

    return x_train, y_train, x_val, y_val, x_test, y_test

def main():

    ####### Option #######
    data_dir = 'all'
    data_type = 'subdural'
    split_size = [5,3,2]
    ######################

    save_dir = './datasets/%s_%s/'%(data_dir, data_type)
    if_not_make(save_dir)
    count_dir = len(listdir(save_dir))+1
    save_dir = save_dir+'dataset_%i'%count_dir
    if_not_make(save_dir)

    data_dir = '../3-Preprocessing/res_%s/%s/3-3/'%(data_dir, data_type)
    img_np = np.load(data_dir + '/img_data.npy')
    label_np = np.load(data_dir + '/label_data.npy')
    print('* Save in [ %s ] '%save_dir)
    
    x_train, y_train, x_val, y_val, x_test, y_test = make_dataset(img_np, label_np, save_dir, split_size)
    
    log = '''
    -- Numpy dataset from [ {d_dir} ][ {d_type} ] --
    * number of dataset : {d_len}
    -- Split dastaset shape --
    * split size = {split_size}
    * x_train, y_train: {x1},{y1}
    * x_val, y_val: {x2},{y2}
    * x_test, y_test: {x3},{y3}
    '''.format(d_dir=data_dir, d_type=data_type, d_len=len(label_np),
               split_size=split_size,
               x1=x_train.shape, y1=y_train.shape, 
               x2=x_val.shape, y2=y_val.shape, 
               x3=x_test.shape, y3=y_test.shape)
    with open(save_dir + '/log.txt', 'a') as log_file:
        log_file.write(log)
    with open(save_dir + '/log.txt', 'w') as log_file:
        log_file.write(log)

    print(log)
if __name__ == '__main__':
    main()   
