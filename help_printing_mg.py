# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 14:12:45 2020

@author: MIT-DGMIF
"""

import os, sys
import numpy as np


types = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any", "normal", "all"]


print_types = '''* 1 : {type1}
* 2 : {type2}
* 3 : {type3}
* 4 : {type4}
* 5 : {type5}
* 6 : {type6}
* 7 : {type7}
* 8 : {type8}'''.format(type1 = types[0], type2 = types[1], type3 = types[2], type4 = types[3], type5 = types[4], type6 = types[5], type7 = types[6], type8 = types[7])


def summary(save_dir, output_list, count=[]):
    outputs = []
    for f in os.listdir(save_dir):
        outputs.append(f.ljust(1))
        
    print('\n* SUMMARY *')
    print('* Output values are saved in [ %s ]'%save_dir)
    print('* Output :', outputs)
    print('  - id_data shape : %s'%(str(output_list[0].shape)))
    print('  - label_data shape : %s'%(str(output_list[1].shape)))
    
    
    if len(output_list) == 3:
        print('  - image file shape : %s'%(str(output_list[2].shape)))
    
    if len(count) == 1:
        print('\n* Counting  label 1 : %i'%(count[0]))
        
    else:
        print('\n* Counting  label 0 : %i   label 1 : %i'%( count[0],  count[1]))
        print('* Ratio is ', (count[1] / count[0]) * 100, '%')
        

def if_not_exit(path):
    if not os.path.exists(path):
        print(path, 'is not exist.')
        exit()

def if_not_make(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
        
        
        
        
