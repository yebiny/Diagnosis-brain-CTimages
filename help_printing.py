import os, sys

types = ["normal", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "any"]
print_types = '''
* 0 : {type0}
* 1 : {type1}
* 2 : {type2}
* 3 : {type3}
* 4 : {type4}
* 5 : {type5}
* 6 : {type6}'''.format(type0=types[0], type1=types[1],type2=types[2],type3=types[3],type4=types[4],type5=types[5],type6=types[6])

def summary(save_dir, output_list):
    outputs = []
    for f in os.listdir(save_dir):
    	outputs.append(f.ljust(1))
    
    print('\n* SUMMARY *')
    print('* Output values are saved in [ %s ]'%save_dir)
    print('* Output :', outputs)
    print('  - id_data shape : %s'%(str(output_list[0].shape)))
    if len(output_list) == 2:
        print('  - label_data shape : %s'%(str(output_list[1].shape)))
    if len(output_list) == 3:
    	print('  - image file shape : %s'%(str(output_list[2].shape)))

def if_not_exit(path):
	if not os.path.exists(path):
		print(path, 'is not exist.')
		exit()
		
def if_not_make(path):
	if not os.path.exists(path):
		os.makedirs(path)
