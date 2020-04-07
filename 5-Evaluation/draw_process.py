from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
mpl.use("Agg")
import os,sys
import pandas as pd

def draw_process(target_dir):
    print(' * Drawing leaning process')
    info = pd.read_csv("../4-Training/results/%s/loss.csv"%(target_dir))
    
    acc =info['accuracy' ]
    loss =info['loss' ]
    vacc=info['val_accuracy']
    vloss=info['val_loss']
    
    x_len = np.arange(len(loss))
    
    # Plot
    plt.plot(x_len, acc, marker='.', c = 'green',label = 'Acc: Train-Set')
    plt.plot(x_len, loss, marker='.', c = 'blue',label = 'Loss: Train-Set')
    
    plt.plot(x_len, vacc,marker='.', c = 'darkorange', label = 'Acc: Valdation-set')
    plt.plot(x_len, vloss,marker='.', c = 'red', label = 'Loss: Valdation-set')
    plt.legend(fontsize=10)
    plt.grid()
    save_dir = './results/%s/'%(target_dir)
    plt.savefig(save_dir+ "loss.png")

def main(target_dir):
    draw_process(target_dir)

if __name__ == '__main__':
        target_dir = str(input("* Results directory: "))
        main(target_dir)
