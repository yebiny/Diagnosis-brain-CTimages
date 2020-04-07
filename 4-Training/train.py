import os, sys
from os import walk, listdir
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

sys.path.append('../')
from help_printing import *
from split_dataset import *
from models import *

def main():

    ############ Setting ######### 
    data_set = 'all_subdural_1'
    batch_size = 256
    ##############################
    model_name = str(input('- Enter Model name :'))
    save_name = str(input('- Enter Save directory name: '))
    epochs=int(input('- Enter epochs: '))
    ##############################
   
    # New model train. 
    if model_name in model_list:
        train(save_name, data_set, model_name, batch_size, epochs)
    # If you run arleady trained model, dataset and batch size are same as trained model.
    elif model_name in os.listdir('./results/'):
        reader = pd.read_csv('./results/%s/info.csv'%model_name)
        data_set = reader.loc[1,'data']
        batch_size = int(reader.loc[3,'data'])
        train(save_name, data_set, model_name, batch_size, epochs)
    else :
        print("No matched model")


def train(save_name, data_set, model_name, batch_size, epochs):
    
    # Setting
    save_dir = './results/%s/'%(save_name)
    if_not_make(save_dir)
    
    x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(data_set)
    
    # Model & Dataset
    if model_name in model_list:
        model = get_model(model_name)(x_train.shape)
    else:
        model = load_model("./results/"+model_name+"/model.hdf5")
    
    model.summary()
    print('* Dataset : ' , data_set)
    print('* Model  : ' , model_name)
    
    # Options
    # * draw model
    keras.utils.plot_model(model, to_file=save_dir+'/model_plot.png', show_shapes=True, show_layer_names=True)
    # * save only best model
    checkpointer = ModelCheckpoint(filepath=save_dir+'/model.hdf5', verbose=1, save_best_only=True)
    # * learning process
    csv_logger = CSVLogger(save_dir + '/loss.csv')
    # * adjust learing rate
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    
    # Compile and Train
    model.compile('adam','binary_crossentropy',metrics=['accuracy'])
    history = model.fit(
        x_train, y_train, 
        validation_data=(x_val, y_val), 
        epochs=epochs,
        batch_size=batch_size,
        callbacks = [checkpointer, csv_logger, reduce_lr]
        )
    
    # Save Log
    data = [save_dir, data_set, model_name, batch_size, epochs]
    data = pd.DataFrame(data)
    data.to_csv(save_dir+'info.csv' ,header=['data'], index=False)


if __name__ == '__main__':
    main()
