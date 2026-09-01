########################################################################################
# Aquire the original data.
# The data was copied from the dataset of my another project, which was downloaded from 
# SCOPe.
########################################################################################

import os

with open('../Datasets/astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa','r') as f:
    lines = f.readlines()

origin_path = '../../../../DeepDesign_New/Data/Datasets/Original/pdb_SCOPe_100/'
origin_list = os.listdir(origin_path)
result_path = '../Datasets/pdb_origin/'

num = 0  
for line in lines:
    if line[0] == '>':
        name = line.strip('\n').split(' ')[0].strip('>') + '.pdb'
        if name in origin_list:
            os.system('cp ' + origin_path + name + ' ' + result_path)
            num += 1
        else:
            print name

print '%d pdb files aquired.'%num 
