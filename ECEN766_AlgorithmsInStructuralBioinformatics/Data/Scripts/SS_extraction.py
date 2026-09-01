import Feature_helper
import os
import pickle

data_path = '../Datasets/'

with open(data_path + 'fold_protein_dict.pickle', 'rb') as handle:
    fold_pro_dict = pickle.load(handle)

pro_list = []
for f in fold_pro_dict:
    pro_list += fold_pro_dict[f]

with open(data_path + 'astral-scopedom-seqres-gd-sel-gs-bib-40-2.07.fa','r') as f:
    lines = f.readlines()
    len_lines = len(lines)

pro_file = open(data_path + 'protein_list','w')
seq_file = open(data_path + 'seq_list','w')
ss_file = open(data_path + 'ss_list','w')
read_check_file = open(data_path + 'read_check','w')
seq_check_file = open(data_path + 'seq_check','w')
chain_check_file = open(data_path + 'chain_check','w')

pro_file.close()
seq_file.close()
ss_file.close()
read_check_file.close()
seq_check_file.close()
chain_check_file.close()

seq_num = 0

for i in range(len_lines):
    line = lines[i]
    if line[0] == '>':
        if i != 0 :
                                      
            if pro in pro_list:       

                seq = seq.upper()
                pdb_file = data_path + 'pdb_origin/' + pro + '.pdb'

                print(pdb_file,chain)

                feature_read =  Feature_helper.ss_info(pdb_file,chain,ss_kind = 3)

                if feature_read[0] != None:

                    seq_extract = feature_read[0]
                    ss = feature_read[1]
    
                    if seq_extract != seq:
                        
                        seq_check_file = open(data_path + 'seq_check','a')

                        seq_check_file.write(pro + '\t' + fold + '\t' + chain + '\n')
                        seq_check_file.write(seq + '\n')
                        seq_check_file.write(seq_extract + '\n')
                        seq_check_file.write('\n')
                         
                        seq_check_file.close()

                    else:
                        pro_file = open(data_path + 'protein_list','a')
                        seq_file = open(data_path + 'seq_list','a')
                        ss_file = open(data_path + 'ss_list','a')

                        pro_file.write(pro + '\t' + fold + '\n')
                        seq_file.write(seq + '\n')
                        ss_file.write(ss + '\n')
                   
                        pro_file.close()
                        seq_file.close()
                        ss_file.close()                    

                        seq_num += 1

                else:
                    read_check_file = open(data_path + 'read_check','a')
                    if len(feature_read) >= 3:
                        read_check_file.write(pro + '\tread_error\t' + str(feature_read[1:]) + '\n')
                    else:
                        read_check_file.write(pro + '\tss_kind_error\n')
                    read_check_file.close()

        pro = line.split(' ')[0].strip('>')    
        fold = line.split(' ')[1]
        fold = '.'.join(fold.split('.')[0:2])
        chain = line.split(' ')[2]
       
        if ',' in chain:
            chain_check_file = open(data_path + 'chain_check','a')
            chain_check_file.write(line)
            chain_check_file.close()
        
        chain = chain[1]
 
        seq = ''
    else:
        seq += line.strip('\n')

if pro in pro_list:

    seq = seq.upper()
    pdb_file = data_path + 'pdb_origin/' + pro + '.pdb'
    feature_read =  Feature_helper.ss_info(pdb_file,chain,ss_kind = 3)

    if feature_read[0] != None:

        seq_extract = feature_read[0]
        ss = feature_read[1]

        if seq_extract != seq:

            seq_check_file = open(data_path + 'seq_check','a')

            seq_check_file.write(pro + '\t' + fold + '\t' + chain + '\n')
            seq_check_file.write(seq + '\n') 
            seq_check_file.write(seq_extract + '\n')
            seq_check_file.write('\n')

            seq_check_file.close()

        else:
 
            pro_file = open(data_path + 'protein_list','a')
            seq_file = open(data_path + 'seq_list','a')
            ss_file = open(data_path + 'ss_list','a')

            pro_file.write(pro + '\t' + fold + '\n')
            seq_file.write(seq + '\n')
            ss_file.write(ss + '\n')

            pro_file.close()
            seq_file.close()
            ss_file.close()
            
            seq_num += 1

    else:
        read_check_file = open(data_path + 'read_check','a')
        if len(feature_read) >= 3:
             read_check_file.write(pro + '\tread_error\t' + str(feature_read[1:]) + '\n')
        elif feature_read[1] == 'ss_kind':
             read_check_file.write(pro + '\tss_kind_error\n')
        else:
             read_check_file.write(pro + '\tresi_dict_error\n')
        read_check_file.close()

print('%d sequences in all.'%seq_num)
