import os, sys
from os import walk, listdir
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model 
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

sys.path.append('../')
from help_printing import *
from make_trainset import *
from models import *

if_not_make('./results')
count = len(listdir('./results'))+1
save_name='run_%i'%(count)


# Option
#####################################################
data_dir = 'all_subdural'
data_set = 'dataset_1'
model_name='base_model'
batch_size=256
epochs=10
#####################################################

# Load Dataset
save_dir = './results/%s/'%(save_name)
if_not_make(save_dir)
load_data= './datasets/%s/%s/'%(data_dir,data_set)
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(load_data)

# Model
if model_name in model_list:
    model = get_model(model_name)(x_train.shape)
else: 
    model = load_model("./results/"+model_name+"/model.hdf5")

model.summary()

# Options
keras.utils.plot_model(model, to_file=save_dir+'/model_plot.png', show_shapes=True, show_layer_names=True)
checkpointer = ModelCheckpoint(filepath=save_dir+'/model.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger(save_dir + '/loss.csv')
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
log= '''
* data directory = {data_dir}
* dataset = {data_set}
* model = {model_name}
* batch_size = {batch_size}
* epochs = {epochs}
'''.format(data_dir=data_dir, data_set=data_set, model_name=model_name, batch_size=batch_size, epochs=epochs)
with open(save_dir + '/log.txt', 'a') as log_file:
    log_file.write(log)
with open(save_dir + '/log.txt', 'w') as log_file:
    log_file.write(log)

print(log)
