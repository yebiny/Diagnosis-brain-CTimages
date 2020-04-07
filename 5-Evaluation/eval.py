import os, sys
import numpy as np
from array import array
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

sys.path.append('../')
from help_printing import *
from draw_process import draw_process
from draw_roc import draw_roc

def main():

    # Training results
    target_dir = str(input("Which directory to evaluate? "))
    target_path = '../4-Training/results/%s'%(target_dir)
    target_info = '%s/info.csv'%(target_path)
    reader = pd.read_csv(target_info)
    
    # Data
    data_path = '../4-Training/'+str(reader.loc[1,'data'])
    x_test = np.load(data_path + 'x_test.npy')
    y_real = np.load(data_path + 'y_test.npy')

    # Model
    model_path=target_path+'/model.hdf5'
    model = tf.keras.models.load_model(model_path)

    # Predict
    print('* Model apply for test set')
    y_score = model.predict(x_test)
    
    acc=0
    for i in range(len(y_score)):
        y_pred = int(round(y_score[i][0]))
        if y_real[i] == y_pred:
            acc = acc+1
    print('[Acc] :' , acc/len(y_score))
   
    y1 = y_real.astype(np.bool)
    y0 = np.logical_not(y1)
    y1_pred = y_score[y1]
    y0_pred = y_score[y0]

    save_dir = './results/%s/'%(target_dir)
    if_not_make(save_dir)
    np.savez(save_dir+'eval.npz', 
             y1 = y1, y0 = y0,
             y1_pred = y1_pred, 
             y0_pred = y0_pred,
             y_real=y_real, 
             y_score=y_score
             )

    # Draw Roc (testset)
    draw_roc(target_dir)
    # Draw Process(traing, validation)
    draw_process(target_dir)
    
if __name__ == '__main__':
    main()
