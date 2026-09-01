from __future__ import print_function
import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from graph import *
# import node2vec
from classify import Classifier, read_node_label
# import line
# import tadw
# from gcn import gcnAPI
# import lle
# import hope
# import lap
import gf
# import sdne
# from grarep import GraRep
import time
import ast

def calc_embedding_from_file(filename_in, filename_out="emb_out.txt"):
    t1 = time.time()
    g = Graph()
    print("Reading...")       

    g.read_edgelist(filename=filename_in, weighted=True,
                    directed=False)
    
    model = gf.GraphFactorization(g, rep_size=128,
                                      epoch=20, learning_rate=0.001, weight_decay=5e-4) # rep_size=128, epoch=5, learning_rate=0.01, weight_decay=5e-4
    t2 = time.time()
    print(t2-t1)

    print("Saving embeddings...")
    model.save_embeddings(filename_out)


def calc_embedding(P_set, N_set, filename_out="emb_out.txt"):
    t1 = time.time()
    g = Graph()
    print("Reading...")


    filename_in = "input.txt"
    with open(filename_in, "w") as f:
        for i in P_set:
            f.write(str(i[0])+" "+str(i[1])+" "+"1"+"\n")
        
        for i in N_set:
            f.write(str(i[0])+" "+str(i[1])+" "+"0"+"\n")
        

    g.read_edgelist(filename=filename_in, weighted=True,
                    directed=False)
    
    model = gf.GraphFactorization(g, rep_size=128,
                                      epoch=20, learning_rate=0.001, weight_decay=5e-4) # rep_size=128, epoch=5, learning_rate=0.01, weight_decay=5e-4
    t2 = time.time()
    print(t2-t1)

    print("Saving embeddings...")
    model.save_embeddings(filename_out)

    name_list_in = [
    'mat_protein_protein_edgelist.txt',
    'Similarity_Matrix_Proteins_edgelist.txt',
    'Similarity_Matrix_Drugs_edgelist.txt',
    'mat_drug_se_edgelist.txt',
    'mat_protein_drug_edgelist.txt',
    'mat_drug_drug_edgelist.txt',
    'mat_drug_disease_edgelist.txt',
    'mat_protein_disease_edgelist.txt']
    data_path = "/scratch/user/rujieyin/OpenNE/data/neodti/"

    # for fname in name_list_in:
    #     calc_embedding_from_file(filename_in = data_path+fname, filename_out=fname[:-12]+"emb.txt")


    tt_ll = []
    with open(filename_out, "r") as f:
        ct = -1
        for line in f:
            if ct == -1:
                ct += 1
                continue
            ll = [float(i) for i in line.strip().split()]
            tt_ll.append(ll)
            ct += 1
    tt_ll = np.array(tt_ll)
    
    tt_ll_copy = np.copy(tt_ll)

    tt_ll = tt_ll[:,1:]

    tt_ll[tt_ll_copy[:, 0].astype(int)] = tt_ll_copy[:, 1:]

    # Average over all the embeddings
    n_drug = 708
    n_protein = 1512

    name_list1 = [
    'Similarity_Matrix_Drugs_emb.txt',
    'mat_drug_se_emb.txt',
    'mat_drug_drug_emb.txt',
    'mat_drug_disease_emb.txt'
    ]

    for fname in name_list1:
        tt_ll1 = []
        with open(fname, "r") as f:
            ct = -1
            for line in f:
                if ct == -1:
                    ct += 1
                    continue
                ll = [float(i) for i in line.strip().split()]
                tt_ll1.append(ll)
                ct += 1
        
        tt_ll1 = np.array(tt_ll1)

        tt_ll_copy = np.copy(tt_ll1)

        tt_ll1 = tt_ll1[:,1:]

        tt_ll1[tt_ll_copy[:, 0].astype(int)] = tt_ll_copy[:, 1:]

        tt_ll[:n_drug] += tt_ll1[:n_drug]




    name_list2 = [
    'mat_protein_protein_emb.txt',
    'Similarity_Matrix_Proteins_emb.txt',
    'mat_protein_drug_emb.txt',
    'mat_protein_disease_emb.txt'
    ]


    for fname in name_list2:
        tt_ll1 = []
        with open(fname, "r") as f:
            ct = -1
            for line in f:
                if ct == -1:
                    ct += 1
                    continue
                ll = [float(i) for i in line.strip().split()]
                tt_ll1.append(ll)
                ct += 1
        
        tt_ll1 = np.array(tt_ll1)

        tt_ll_copy = np.copy(tt_ll1)

        tt_ll1 = tt_ll1[:,1:]

        tt_ll1[tt_ll_copy[:, 0].astype(int)] = tt_ll_copy[:, 1:]

        tt_ll[n_drug:] += tt_ll1[:n_protein]
    

    name_list3 = [
        'mat_protein_drug_emb.txt'
    ]

    for fname in name_list3:
        tt_ll1 = []
        with open(fname, "r") as f:
            ct = -1
            for line in f:
                if ct == -1:
                    ct += 1
                    continue
                ll = [float(i) for i in line.strip().split()]
                tt_ll1.append(ll)
                ct += 1
        
        tt_ll1 = np.array(tt_ll1)

        tt_ll_copy = np.copy(tt_ll1)

        tt_ll1 = tt_ll1[:,1:]

        tt_ll1[tt_ll_copy[:, 0].astype(int)] = tt_ll_copy[:, 1:]

        tt_ll[:n_drug] += tt_ll1[n_protein:]

    tt_ll[:n_drug] /= 6
    tt_ll[n_drug:] /= 5



    def get_embedding(u):
        return tt_ll[u]

    return get_embedding

