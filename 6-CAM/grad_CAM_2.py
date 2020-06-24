# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:06:05 2020

@author: MIT-DGMIF
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

class tuple_dict(dict):
    """
    Custom dict object accepting tuples as 'keys' and unpacking their elements
    to get dict items corresponding to keys contained within the tuple.
    """
    def __init__(self, *args):
        dict.__init__(self, args)

    def __getitem__(self, i):
        if type(i) == tuple:
            lst = []
            for n in i:
                lst.append(dict.__getitem__(self, n))
            return lst
        elif type(i) == str:
            return dict.__getitem__(self, i)


def get_outputs_at_each_layer(model, input_image, layer_type):
    """
    Returns outputs and gradients of the score function with respect to
    those layer-wise outputs. Restricts layer type for gradient calculation
    to optimize derivation with respect to feature maps of convolutional layers.
    This function works when eager mode in TensorFlow is enabled.
    :param model: tf.keras.models Model or Sequential object
    :param input_image: input image as a tf.Tensor
    :param layer_type: layer class e.g. tf.keras.layers.Conv2D
    :return: tuple of ouptuts of respective layers and gradients associated with them
    """
    with tf.GradientTape() as tape:
        outputs = tuple_dict()  # custom dict object
        outputs['input_image'] = input_image  # initialize first input
        # this is because the first layer doesn't take input from any other layer below
        restricted_outputs = []

        nodes_by_depth_keys = list(model._nodes_by_depth.keys())
        nodes_by_depth_keys.sort(reverse=True)  # this is the proper order of executing ops
        for k in nodes_by_depth_keys:
            nodes = model._nodes_by_depth[k]
            for n in nodes:
                config = n.get_config()  # returns a dict
                # inbound_node in config gets the input node
                # outbound_node in config points to the operation
                if type(config['outbound_layer']) == list:  # convert lists to tuples
                    obl = tuple(config['outbound_layer'])
                else:
                    obl = config['outbound_layer']
                if type(config['inbound_layers']) == list:
                    if len(config['inbound_layers']) == 0:
                        # we expect only the first layer inbound nodes to be non-existent
                        ibl = 'input_image'
                    else:
                        ibl = tuple(config['inbound_layers'])
                else:
                    ibl = config['inbound_layers']
                out = model.get_layer(obl)(outputs[ibl])  # magic happens here
                # we call each outbound node with its inbound nodes...
                outputs[obl] = out  # ...and we append it back to the dict containing outputs
                # keys in the dict are names of layers/nodes, which allows this loop
                # to get them anytime multiple inputs are needed
                if isinstance(model.get_layer(obl), layer_type):
                    # we return only those layers that we want to see (conv)
                    restricted_outputs.append(outputs[obl])
    gradients = tape.gradient(out, restricted_outputs)  # record 
    return restricted_outputs, gradients


def grad_cam(img, model, dim, return_switch=None, watch_layer_instances=tf.keras.layers.Conv2D):
    """
    Grad-CAM visualization function.
    :param image: path to image as a string
    :param model: tf.keras.models Model or Sequential object
    :param image_dims: tuple specifying size of the output photo
    :param return_switch: 'gradients', 'maps', 'both', 'upsampled' or 'summed'
                          switches output of the function to return gradients,
                          feature maps, both gradients and feature maps,
                          upsampled feature maps or summed feature maps respectively
    :param watch_layer_instances: single class or tuple of classes of layers to watch
                                  this is useful for tracking special offshoots of conv layers
    :return: values as specified by return_switch
    This function produces Grad-CAM plots as a side effect
    """
  
    
    print('input shape:',img.shape)
    image_dims = img.shape[:2]
    
    
    Image.fromarray(img.astype(np.uint8)).save("test.png")
    im_tf = Image.open('test.png')
    
    A_k, dy_dA_k = get_outputs_at_each_layer(model, tf.cast(np.expand_dims(img, axis=0), tf.float32), watch_layer_instances)
    L_c = [tf.keras.layers.ReLU()(tf.math.reduce_sum(tf.math.multiply(dy_dA_k[i], A_k[i]), axis=(3))) for i, _ in enumerate(dy_dA_k)]
    up_all = [np.array(Image.fromarray(i.numpy()[0, :, :]).resize(image_dims, resample=Image.BILINEAR)) for i in L_c]
    summed_maps = tf.keras.layers.ReLU()(np.sum(up_all, axis=0))
    np.save('gradient.npy', summed_maps.numpy())
    
    if dim == 1: img = img.reshape(image_dims)
    
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('Gradient with 0.9 alpha')
    plt.axis('off')
    plt_im1 = plt.imshow(im_tf, interpolation='bilinear')
    plt_im2 = plt.imshow(summed_maps.numpy(), cmap='magma', alpha=.95, interpolation='nearest')

    ax2 = fig.add_subplot(132)
    ax2.title.set_text('Gradient with 0.6 alpha')
    plt.axis('off')
    plt_im1 = plt.imshow(im_tf, interpolation='bilinear')
    plt_im2 = plt.imshow(summed_maps.numpy(), cmap='magma', alpha=.6, interpolation='nearest')
    
    
    ax2 = fig.add_subplot(133)
    ax2.title.set_text('Original image')
    plt.axis('off')
    plt_im1 = plt.imshow(im_tf, interpolation='bilinear', cmap = 'bone')
    #plt.subplots_adjust(hspace = 4, wspace = 0.3, right = 0.8, left = - 0.8)
    plt.savefig('cam.png', dpi = 300)
    plt.show()
    
    
    if isinstance(return_switch, str):
        if (return_switch == 'gradients'):
            return dy_dA_k
        elif (return_switch == 'both'):
            return A_k, dy_dA_k
        elif (return_switch == 'maps'):
            return A_k
        elif (return_switch == 'upsampled'):
            return up_all
        elif (return_switch == 'summed'):
            return summed_maps
        else:
            return None
    elif (return_switch is None):
        return None
    else:
        raise RuntimeError('Invalid return value switch!')


    
def show_cam_img(idx, x_test, y_test, y_pred):
    print('* real y : ', y_test[idx], '* predict y :' , y_pred[idx])
    grad_cam(x_test[idx], model, 3)
 
#%%
data_path = 'E:/Dataset/rsna-intracranial-hemorrhage-detection/brain_diagnosis/6-CAM/'

x_test = np.load('%s/X_test.npy'%data_path)
y_test = np.load('%s/y_test.npy'%data_path)

x_test.shape, y_test.shape 

#%%

import efficientnet.tfkeras as efn
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras import utils, models, layers, optimizers
from tensorflow.keras.models import Model, load_model, Sequential

base_model = efn.EfficientNetB7(weights = 'imagenet', include_top = False, pooling = 'avg', input_shape = (256, 256, 3))
x = base_model.output
y = Dropout(0.5)(x)
y = Dense(1024, activation='relu')(y)
y = Dense(5, activation = 'sigmoid')(y)
model = Model(base_model.input, y)
model.summary()

#%%
model.load_weights('E:/Dataset/rsna-intracranial-hemorrhage-detection/brain_diagnosis/5-Results/best_model/effb7/weight_Classification_200616_Adam_effb7.hdf5')


#%%
y_pred = model.predict(x_test[:10])
show_cam_img(3, x_test, y_test, y_pred)





#%%%

import os, glob
import cv2 as cv
import pandas as pd
import seaborn as sns
# 분류 대상 카테고리 선택하기 

categories = ['epidural','intraparenchymal','intraventricular', 'subarachnoid', 'subdural']
nb_classes = len(categories)
# 이미지 크기 지정 
image_w = 256
image_h = 256
pixels = image_w * image_h * 3

predictions = y_pred
pred =  predictions.astype("float16")

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    norm_img = cv.normalize(img.astype(np.uint8), None, 0, 255, cv.NORM_MINMAX)
    plt.imshow(norm_img, cmap=plt.cm.bone)
    plt.title('Intracranial Hemorrahage Image Number: %d' % i, fontsize = 20)
    tmp = []
    predicted_label = []
    for idx in range(len(predictions_array)):
        if predictions_array[idx] > 0.05:
            tmp.append('1')
        else:
            tmp.append('0')
        
        predicted_label.append(tmp[idx])
    predicted_label = np.array(predicted_label).astype(np.uint8)
    
    if list(predicted_label>0.05) == list(true_label>0):
        color = 'red'
    else:
        color = 'black'
        
    pred_bool = predicted_label > 0
    label_bool = true_label > 0    
    categor = np.array(categories)
    if list(label_bool).count(False) == 5:
        plt.xlabel("{}".format(str("['Normal']"), color = color))
    else:
        plt.xlabel("{}".format(str(categor[label_bool]), color = color))
    #plt.xlabel("{} {}%".format(str(categor[pred_bool]), str(100 * (predictions_array)), color = color))
    
'''    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    
    categories = ['epidural','intraparenchymal','intraventricular', 'subarachnoid', 'subdural']
    
    
    df = pd.DataFrame({'GT' : true_label, 'Pred' : predictions_array, '[Hemorrahage sub-type]' : categories})
    df = df.rename_axis('G/P')
    

    #thisplot_1[predicted_label[0]].set_color('red')
    #pnt = np.array(np.where(true_label!=predicted_label))
    #for ii in pnt[0]:
        #thisplot[ii].set_color('pink')
    
    #df['color'] = np.where(true_label!=predicted_label, 'red', np.where(true_label == predicted_label, 'green', 'blue'))
    thisplot = df.plot(kind = 'bar', rot = 0, x = '[Hemorrahage sub-type]', color = ['slateblue', 'blue'])
'''
'''
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    #ax = plt.subplot()
    
    t = 2 # Number of dataset
    d = 5 # Number of sets of bars
    w = 0.5 # Width of each bar
    nb_cl_x = [t*element + w*1 for element in range(d)]
    nb_cl_y = [t*element + w*2 for element in range(d)]
    
    thisplot_1 = plt.bar(nb_cl_x, true_label, label = 'GT', color = "b")
    thisplot_2 = plt.bar(nb_cl_y, predictions_array, label = 'Pred', color = "r")
    plt.legend()
    plt.ylim([0, 1])
    plt.xticks([])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlabel(categories, ha = 'center', fontsize = 6)
    #plt.xlabel("{}    {}    {}    {}    {}".format(str(categories[0]), str(categories[1]), str(categories[2]), str(categories[3]), str(categories[4])))
    tmp = []
    predicted_label = []
    for idx in range(len(predictions_array)):
        if predictions_array[idx] > 0.1:
            tmp.append('1')
        else:
            tmp.append('0')
        
        predicted_label.append(tmp[idx])
    predicted_label = np.array(predicted_label).astype(np.uint8)
    
    #thisplot_1[predicted_label[0]].set_color('red')
    pnt = np.array(np.where(true_label!=predicted_label))
    for ii in pnt[0]:
        thisplot_2[ii].set_color('pink')
'''
    
def compute_pos(xticks, width, i , models):
    index = np.arange(len(xticks))
    n = len(models)
    correction = i - 0.5 * (n-1)
    return index + width * correction

def present_height(ax, bar):
    for rect in bar:
        height = rect.get_height()
        posx = rect.get_x() + rect.get_width() * 0.5
        posy = height * 1.01
        ax.text(posx, posy, '%.3f' % height, rotation = 90, ha = 'center', va = 'bottom')

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    
    #### 1. bar plot으로 나타낼 데이터 입력
    models = ['Ground Truth', 'Prediction']
    xticks = ['epidural','intraparenchymal','intraventricular', 'subarachnoid', 'subdural']
    data = {'Ground Truth' : list(true_label), 'Prediction' : list(predictions_array)}
    
    
    #### 2. matplotlib의 figure 및 axis 설정
    fig, ax = plt.subplots(1,1,figsize=(18,8)) # 1x1 figure matrix 생성, 가로(7인치)x세로(5인치) 크기지정
    colors = ['midnightblue', 'darkred']
    width = 0.15
    
    
    #### 3. bar 그리기
    for ix, model in enumerate(models):
        pos = compute_pos(xticks, width, ix, models)
        bar = ax.bar(pos, data[model], width=width*0.95, label=model, color=colors[ix])
        present_height(ax, bar) # bar높이 출력
        
        
    #### 4. x축 세부설정
    ax.set_xticks(range(len(xticks)))
    ax.set_xticklabels(xticks, fontsize=18)	
    ax.set_xlabel('Sub-type', fontsize=30)
    
    #### 5. y축 세부설정
    ax.set_ylim([0.0, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_tick_params(labelsize=10)
    ax.set_ylabel('Prediction Accuracy', fontsize=30)
    
    
    #### 6. 범례 나타내기
    ax.legend(loc='upper left', shadow=True, ncol=1)
    
    
    #### 7. 보조선(눈금선) 나타내기
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
    
    
    tmp = []
    predicted_label = []
    for idx in range(len(predictions_array)):
        if predictions_array[idx] > 0.1:
            tmp.append('1')
        else:
            tmp.append('0')
        
        predicted_label.append(tmp[idx])
    predicted_label = np.array(predicted_label).astype(np.uint8)
    
    #thisplot_1[predicted_label[0]].set_color('red')
    pnt = np.array(np.where(true_label!=predicted_label))
    #0:[5] 1:[6] 2:[7] 3:[8] 4:[9]
    for ii in pnt[0]:
        ax.get_children()[int(ii) + 5].set_color('mistyrose')    
    
    plt.title('Intracranial Hemorrahage Image Number: %d' % i, fontsize = 20)
    #### 8. 그래프 저장하고 출력하기
    plt.tight_layout()
    plt.savefig('ex_barplot.png', format='png', dpi=300)
    plt.show()
    

    
num_rows = 1
num_cols = 2
start_point =3
#num_images = num_rows * num_cols
plt.figure(figsize = (2 * 2 * num_cols, 2 * num_rows))

#for i in range(num_images):
i = 0
plt.subplot(num_rows, num_cols, i + 1)
plot_image(i + start_point, predictions, y_test, x_test)
plt.subplot(num_rows, num_cols, i + 2)
plot_image(i + start_point, predictions, y_test, x_test)
gd = np.load('gradient.npy')
plt.imshow(gd, cmap='jet', alpha=.2, interpolation='nearest')
plt.show()
#for i in range(num_images):
#plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
plot_value_array(i + start_point, predictions, y_test)
plt.show()





