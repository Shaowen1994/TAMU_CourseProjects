###################################################################################
# Contain functions and classes to help to extract secondary structure information
###################################################################################

import sys
import numpy as np
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import string

alphabet_string = string.ascii_uppercase

def remove(string,char):
    '''
    Remove the space in a string.
    '''
    string_char = [i for i in string.split(char) if i != '']
    return string_char[0]

def read_pdb(pdb_file):
    '''
    Extract the residue information inside a pdb file.
    '''
    protein_dict = {}
    with open(pdb_file,'r') as p_file:
        lines = p_file.readlines()
        for line in lines:
            if line[0:4] == 'ATOM':
                atom = remove(line[13:17],' ')
                resi = line[17:20]
                chain = line[21]
                index = remove(line[22:27],' ')
                x = float(remove(line[30:38],' '))
                y = float(remove(line[38:46],' '))
                z = float(remove(line[46:54],' '))
        ############ Judge whether a new chain begins. ########################      
                if not chain in protein_dict.keys():
                    protein_dict[chain] = {}
        ############ Save the sequence infomation. ######################## 
                if not index in protein_dict[chain].keys():
                    protein_dict[chain][index] = {'resi':resi}
                elif resi != protein_dict[chain][index]['resi']:
                    print(pdb_file)
                    print('PDB read error! The residue kind of resi %s is not consistent!'%index)
                    return index
                protein_dict[chain][index][atom] = [x,y,z]  
    return protein_dict

def index_split(index):
    '''
    Split the number and the alphabet part in the index.
    '''
    if index[-1] in alphabet_string:
        value = int(index[:-1])
        foot =  index[-1]
    else:
        value = int(index)
        foot = ' '
    return value,foot,index   

class SS_extraction:
   
    def __init__(self,pdb_file,AA_kind='common'):
        if AA_kind ==  'common':
            self.AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X'} 
        else:
            self.AA_dict = {'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V','UNK':'X','SEC':'B'}
        
        self.ss_dict_8_3 = {'H':'H','G':'H','I':'H','E':'E','B':'E','S':'C','T':'C','-':'C','C':'C','X':'X','x':'x','M':'M'} # 8-classes to 3-classes and 3 to 3

        self.pdb_file = pdb_file
        self.protein_dict = read_pdb(pdb_file)
        
        try:
            dssp_dict = dssp_dict_from_pdb_file(pdb_file)[0]
            self.dssp_read = True
        except:
            self.dssp_read = False

        self.pdb_read = (type(self.protein_dict) == dict) 

        if self.pdb_read and self.dssp_read:

            self.Seq_dict = {}
            self.SS_dict_8 = {}
            self.SS_dict_3 = {}

            for chain in self.protein_dict.keys():
                Complete_Seq = ''
                Complete_SS_8 = ''
                Complete_SS_3 = ''

                index_info = sorted([index_split(i) for i in self.protein_dict[chain].keys()],key = lambda x: x[0])

                indv_pre = index_info[0][0] - 1

                resi_dict_problem = False

                for index_value, index_foot, index in index_info:

                    Complete_Seq += 'x'*(index_value - indv_pre - 1)
                    Complete_SS_8 += 'x'*(index_value - indv_pre - 1)
                    Complete_SS_3 += 'x'*(index_value - indv_pre - 1)

                    dssp_key = (chain,(' ', index_value, index_foot))
                    
                    if self.protein_dict[chain][index]['resi'] in self.AA_dict.keys():
                        resi_abbre = self.AA_dict[self.protein_dict[chain][index]['resi']]
                    else:
                        resi_dict_problem = True
                        break 

                    if dssp_key in dssp_dict.keys():

                        resi_ss = dssp_dict[dssp_key][1]
                        self.protein_dict[chain][index]['SeconStru'] = resi_ss

                        if resi_abbre != dssp_dict[dssp_key][0]:
                            print('Residue Error! %s and %s do not match!'%(self.protein_dict[chain][index]['resi'],dssp_dict[dssp_key][0]))
                        else:
                            self.protein_dict[chain][index]['AminoAci'] = dssp_dict[dssp_key][0]

                    else:
                        resi_ss = 'M'

                    Complete_Seq += resi_abbre
                    Complete_SS_8 += resi_ss
                    Complete_SS_3 += self.ss_dict_8_3[resi_ss]
                 
                    indv_pre = index_value
                
                if resi_dict_problem:
                    self.Seq_dict[chain] = None
                    self.SS_dict_8[chain] = None
                    self.SS_dict_3[chain] = None
                else:
                    self.Seq_dict[chain] = Complete_Seq
                    self.SS_dict_8[chain] = Complete_SS_8
                    self.SS_dict_3[chain] = Complete_SS_3

def ss_info(pdb_file,chain,ss_kind = 3):
    if not ss_kind in [3,8]:
        print('ss_kind cannot be %s'%ss_kind)
        return None,'ss_kind'
    else:
        info = SS_extraction(pdb_file)
        if info.pdb_read and info.dssp_read:
            seq = info.Seq_dict[chain]
            if ss_kind == 3:
                ss = info.SS_dict_3[chain]
            elif ss_kind == 8:
                ss = info.SS_dict_3[chain]
            return seq,ss
        elif info.pdb_read:
            print('DSSP Read Error!')
            return None,info.pdb_read,info.dssp_read
        else:
            print('PDB Read Error!')
            return None,info.protein_dict,info.dssp_read
            


