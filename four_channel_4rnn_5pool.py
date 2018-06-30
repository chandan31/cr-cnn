# coding=utf-8
'''
@author: ***
'''

import sys
sys.path.append('../data/')
sys.path.append('../util/')
sys.path.append('../Nets/')
sys.path.append('../nn/')

from SemEval2010 import load_rev_direc_samples_test, load_rev_direc_labels_test,\
    load_rev_valid_aug18class_samples_train, load_rev_valid_aug18class_labels_train, \
    load_wrd_vec_dic_v2, lst_2_dic, load_emd_lst_v2, WordNet_44_categories, \
    POS_15_categories, GR_19_categories
from file_io import get_curr_file_name_no_pfix
# from nn import gl
# from nn import Activation
# from nn import InitParam as init
import gl, Activation, InitParam as init
import Layers as Lay
import Connections as Con
import os
import numpy as np
import cPickle  # @UnusedImport
import struct

##############################
# hyper_param
#print 'Hello! Anybody in there?'
num_out = 19   # No of outputs
num_hid = 100  # Final Hidden layer size 

emb_self  = 200   # its a four layer DRNN whats self ?
emb_rec_1 = 200
emb_rec_2 = 200
emb_rec_3 = 200
emb_rec_4 = 200

pos_self  = 50
pos_rec_1 = 50
pos_rec_2 = 50
pos_rec_3 = 50
pos_rec_4 = 50

wn_self  = 50
wn_rec_1 = 50
wn_rec_2 = 50
wn_rec_3 = 50
wn_rec_4 = 50

gr_self  = 50
gr_rec_1 = 50
gr_rec_2 = 50
gr_rec_3 = 50
gr_rec_4 = 50

##############################
# Load data
dir1 = "../Nets"
dir2 = "/" + get_curr_file_name_no_pfix()
dir3 = "/" + "e" + str(emb_self) + "_" + "h" + str(emb_rec_1) + \
       "_POSe" + str(pos_self) + "_WNe" + str(wn_self) + \
       "_GRe" + str(gr_self) + "_Hid" + str(num_hid)

path_raw_data = dir1 + "/"
path_train = dir1 + dir2 + dir3 + "/train/"
#print path_train
#exit(0)
path_valid = dir1 + dir2 + dir3 + "/valid/"
path_test  = dir1 + dir2 + dir3 + "/test/"
path_join  = dir1 + dir2 + dir3 + "/join/"

if not os.path.exists(dir1 + dir2):
    os.mkdir(dir1 + dir2)
if not os.path.exists(dir1 + dir2 + dir3):
    os.mkdir(dir1 + dir2 + dir3)
if not os.path.exists(path_train):
    os.mkdir(path_train)
if not os.path.exists(path_test):
    os.mkdir(path_test)
if not os.path.exists(path_valid):
    os.mkdir(path_valid)
if not os.path.exists(path_join):
    os.mkdir(path_join)

    

file_train = "sem_train_8000.txt"
file_test = "sem_test_2717.txt"

vocab_dic  = load_wrd_vec_dic_v2(path_raw_data)
emb_lst    = load_emd_lst_v2(path_raw_data)

# Ptn_num = 20
wn_num  = 50
pos_num = 15
gr_num  = 19
wn_dic  = lst_2_dic(WordNet_44_categories)
pos_dic = lst_2_dic(POS_15_categories)
gr_dic  = lst_2_dic(GR_19_categories)

##############################
Biases  = np.array([])
Weights = np.array([])

Biases, B_emb_self = init.InitParam(Biases, newWeights=emb_lst)
#print Biases.shape
#exit(0)
Biases, B_pos_self = init.InitParam(Biases, num=pos_self*(pos_num+1))
#print 'Yup'
#print  
Biases, B_wn_self  = init.InitParam(Biases, num=wn_self*(wn_num+1))
#print Biases.shape
Biases, B_gr_self  = init.InitParam(Biases, num=gr_self*(gr_num+1))
#print 'What the hell!'
#print "Biases:", len(Biases)
# For RNN
Biases, B_emb_rec_1 = init.InitParam(Biases, num=emb_rec_1)
#print Biases.shape  

Biases, B_pos_rec_1 = init.InitParam(Biases, num=pos_rec_1)
#print Biases[:10]
#exit(0)
Biases, B_wn_rec_1  = init.InitParam(Biases, num=wn_rec_1)
Biases, B_gr_rec_1  = init.InitParam(Biases, num=gr_rec_1)

Biases, B_emb_rec_2 = init.InitParam(Biases, num=emb_rec_2)
Biases, B_pos_rec_2 = init.InitParam(Biases, num=pos_rec_2)
Biases, B_wn_rec_2  = init.InitParam(Biases, num=wn_rec_2)
Biases, B_gr_rec_2  = init.InitParam(Biases, num=gr_rec_2)

Biases, B_emb_rec_3 = init.InitParam(Biases, num=emb_rec_3)
Biases, B_pos_rec_3 = init.InitParam(Biases, num=pos_rec_3)
Biases, B_wn_rec_3  = init.InitParam(Biases, num=wn_rec_3)
Biases, B_gr_rec_3  = init.InitParam(Biases, num=gr_rec_3)

Biases, B_emb_rec_4 = init.InitParam(Biases, num=emb_rec_4)
Biases, B_pos_rec_4 = init.InitParam(Biases, num=pos_rec_4)
Biases, B_wn_rec_4  = init.InitParam(Biases, num=wn_rec_4)
Biases, B_gr_rec_4  = init.InitParam(Biases, num=gr_rec_4)

Biases, B_hid    = init.InitParam(Biases, num=num_hid)
#
Biases, B_out    = init.InitParam(Biases, num=num_out)
#print Biases.shape, len(B_out)
#exit(0)
###### For RNN
# 1st rnn
Weights, W_emb_self_01   = init.InitMatrixParam(Weights, num=1, n_in=emb_self, n_out=emb_rec_1)

Weights, W_pos_self_01   = init.InitMatrixParam(Weights, num=1, n_in=pos_self, n_out=pos_rec_1)

Weights, W_wn_self_01    = init.InitMatrixParam(Weights, num=1, n_in=wn_self,  n_out=wn_rec_1)

Weights, W_gr_self_01    = init.InitMatrixParam(Weights, num=1, n_in=gr_self,  n_out=gr_rec_1)



Weights, W_emb_rec_11    = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_1,  n_out=emb_rec_1)
Weights, W_pos_rec_11    = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_1,  n_out=pos_rec_1)
Weights, W_wn_rec_11     = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_1,   n_out=wn_rec_1)
Weights, W_gr_rec_11     = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_1,   n_out=gr_rec_1)


# 2nd rnn
Weights, W_emb_self_12  = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_1, n_out=emb_rec_2)
Weights, W_pos_self_12  = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_1, n_out=pos_rec_2)
Weights, W_wn_self_12   = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_1,  n_out=wn_rec_2)
Weights, W_gr_self_12   = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_1,  n_out=gr_rec_2)


Weights, W_emb_rec_22   = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_2,  n_out=emb_rec_2)
Weights, W_pos_rec_22   = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_2,  n_out=pos_rec_2)
Weights, W_wn_rec_22    = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_2,   n_out=wn_rec_2)
Weights, W_gr_rec_22    = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_2,   n_out=gr_rec_2)

Weights, W_emb_rec_12   = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_1,  n_out=emb_rec_2)
Weights, W_pos_rec_12   = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_1,  n_out=pos_rec_2)
Weights, W_wn_rec_12    = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_1,   n_out=wn_rec_2)
Weights, W_gr_rec_12    = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_1,   n_out=gr_rec_2)

# 3rd rnn
Weights, W_emb_self_23  = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_2, n_out=emb_rec_3)
Weights, W_pos_self_23  = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_2, n_out=pos_rec_3)
Weights, W_wn_self_23   = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_2,  n_out=wn_rec_3)
Weights, W_gr_self_23   = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_2,  n_out=gr_rec_3)

Weights, W_emb_rec_33   = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_3,  n_out=emb_rec_3)
Weights, W_pos_rec_33   = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_3,  n_out=pos_rec_3)
Weights, W_wn_rec_33    = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_3,   n_out=wn_rec_3)
Weights, W_gr_rec_33    = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_3,   n_out=gr_rec_3)

Weights, W_emb_rec_23   = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_2,  n_out=emb_rec_3)
Weights, W_pos_rec_23   = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_2,  n_out=pos_rec_3)
Weights, W_wn_rec_23    = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_2,   n_out=wn_rec_3)
Weights, W_gr_rec_23    = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_2,   n_out=gr_rec_3)

# 4th rnn
Weights, W_emb_self_34  = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_3, n_out=emb_rec_4)
Weights, W_pos_self_34  = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_3, n_out=pos_rec_4)
Weights, W_wn_self_34   = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_3,  n_out=wn_rec_4)
Weights, W_gr_self_34   = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_3,  n_out=gr_rec_4)

Weights, W_emb_rec_44   = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_4,  n_out=emb_rec_4)
Weights, W_pos_rec_44   = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_4,  n_out=pos_rec_4)
Weights, W_wn_rec_44    = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_4,   n_out=wn_rec_4)
Weights, W_gr_rec_44    = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_4,   n_out=gr_rec_4)

Weights, W_emb_rec_34   = init.InitMatrixParam(Weights, num=1, n_in=emb_rec_3,  n_out=emb_rec_4)
Weights, W_pos_rec_34   = init.InitMatrixParam(Weights, num=1, n_in=pos_rec_3,  n_out=pos_rec_4)
Weights, W_wn_rec_34    = init.InitMatrixParam(Weights, num=1, n_in=wn_rec_3,   n_out=wn_rec_4)
Weights, W_gr_rec_34    = init.InitMatrixParam(Weights, num=1, n_in=gr_rec_3,   n_out=gr_rec_4)
#print Weights.shape, len(W_emb_rec_11)
#exit(0)
# hidden layer
Weights, W_emb_hid_0   = init.InitMatrixParam(Weights, num=2, n_in=emb_self,  n_out=num_hid)
Weights, W_pos_hid_0   = init.InitMatrixParam(Weights, num=2, n_in=pos_self,  n_out=num_hid)
Weights, W_wn_hid_0    = init.InitMatrixParam(Weights, num=2, n_in=wn_self,   n_out=num_hid)
Weights, W_gr_hid_0    = init.InitMatrixParam(Weights, num=2, n_in=gr_self,   n_out=num_hid)

Weights, W_emb_hid_1   = init.InitMatrixParam(Weights, num=2, n_in=emb_rec_1,  n_out=num_hid)
Weights, W_pos_hid_1   = init.InitMatrixParam(Weights, num=2, n_in=pos_rec_1,  n_out=num_hid)
Weights, W_wn_hid_1    = init.InitMatrixParam(Weights, num=2, n_in=wn_rec_1,   n_out=num_hid)
Weights, W_gr_hid_1    = init.InitMatrixParam(Weights, num=2, n_in=gr_rec_1,   n_out=num_hid)

Weights, W_emb_hid_2   = init.InitMatrixParam(Weights, num=2, n_in=emb_rec_2,  n_out=num_hid)
Weights, W_pos_hid_2   = init.InitMatrixParam(Weights, num=2, n_in=pos_rec_2,  n_out=num_hid)
Weights, W_wn_hid_2    = init.InitMatrixParam(Weights, num=2, n_in=wn_rec_2,   n_out=num_hid)
Weights, W_gr_hid_2    = init.InitMatrixParam(Weights, num=2, n_in=gr_rec_2,   n_out=num_hid)

Weights, W_emb_hid_3   = init.InitMatrixParam(Weights, num=2, n_in=emb_rec_3,  n_out=num_hid)
Weights, W_pos_hid_3   = init.InitMatrixParam(Weights, num=2, n_in=pos_rec_3,  n_out=num_hid)
Weights, W_wn_hid_3    = init.InitMatrixParam(Weights, num=2, n_in=wn_rec_3,   n_out=num_hid)
Weights, W_gr_hid_3    = init.InitMatrixParam(Weights, num=2, n_in=gr_rec_3,   n_out=num_hid)

Weights, W_emb_hid_4   = init.InitMatrixParam(Weights, num=2, n_in=emb_rec_4,  n_out=num_hid)
Weights, W_pos_hid_4   = init.InitMatrixParam(Weights, num=2, n_in=pos_rec_4,  n_out=num_hid)
Weights, W_wn_hid_4    = init.InitMatrixParam(Weights, num=2, n_in=wn_rec_4,   n_out=num_hid)
Weights, W_gr_hid_4    = init.InitMatrixParam(Weights, num=2, n_in=gr_rec_4,   n_out=num_hid)


##### output
Weights, W_hid_out  = init.InitMatrixParam(Weights, num=1, n_in=num_hid,  n_out=num_out)


def path_4rnn(item_idx, item_type, item_dic, item_path,
             layers, layer_off=0, layer_space=5, item_self=200,
             item_rec_1=200, item_rec_2=200, item_rec_3=200, item_rec_4=200,
             B_item_rec_1=None,   B_item_rec_2=None,  B_item_rec_3=None, B_item_rec_4=None,
             W_item_self_01=None, W_item_rec_11=None,
             W_item_self_12=None, W_item_rec_22=None, W_item_rec_12=None,
             W_item_self_23=None, W_item_rec_33=None, W_item_rec_23=None,
             W_item_self_34=None, W_item_rec_44=None, W_item_rec_34=None):
    """
    """
    print 'path_4rnn'
    #print len(item_path)
    #exit(0)
    for i in xrange(len(item_path)):
        item = item_path[i]
        #print item 
        if item in item_dic:
            idx = item_dic[item][0]
        else:
            idx = 0

        # idx 0: embedding node
        B_idx = item_idx[idx*item_self : (idx+1)*item_self]
        item_layer = Lay.layer(item, B_idx, item_self)
        item_layer.act = item_type
        layers.append(item_layer)

        # idx 1: hidden node 1
        rec_layer_1 = Lay.layer("rec_1_" + layers[layer_off+i].name, B_item_rec_1, item_rec_1)
        rec_layer_1.act = "ReLU"
        layers.append(rec_layer_1)

        # idx 2: hidden node 2
        rec_layer_2 = Lay.layer("rec_2_" + layers[layer_off+i].name, B_item_rec_2, item_rec_2)
        rec_layer_2.act = "ReLU"
        layers.append(rec_layer_2)

        # idx 3: hidden node 3
        rec_layer_3 = Lay.layer("rec_3_" + layers[layer_off+i].name, B_item_rec_3, item_rec_3)
        rec_layer_3.act = "ReLU"
        layers.append(rec_layer_3)

        # idx 4: hidden node 4
        rec_layer_4 = Lay.layer("rec_4_" + layers[layer_off+i].name, B_item_rec_4, item_rec_4)
        rec_layer_4.act = "ReLU"
        layers.append(rec_layer_4)

        if i == 0:
            Con.connection(layers[layer_off],   layers[layer_off+1], item_self,  item_rec_1, W_item_self_01)
            Con.connection(layers[layer_off+1], layers[layer_off+2], item_rec_1, item_rec_2, W_item_self_12)
            Con.connection(layers[layer_off+2], layers[layer_off+3], item_rec_2, item_rec_3, W_item_self_23)
            Con.connection(layers[layer_off+3], layers[layer_off+4], item_rec_3, item_rec_4, W_item_self_34)
        else:
            # recurrent axis
            Con.connection(layers[layer_off+i*layer_space],   layers[layer_off+i*layer_space+1],
                           item_self,  item_rec_1, W_item_self_01)
            Con.connection(layers[layer_off+i*layer_space+1], layers[layer_off+i*layer_space+2],
                           item_rec_1, item_rec_2, W_item_self_12)
            Con.connection(layers[layer_off+i*layer_space+2], layers[layer_off+i*layer_space+3],
                           item_rec_2, item_rec_3, W_item_self_23)
            Con.connection(layers[layer_off+i*layer_space+3], layers[layer_off+i*layer_space+4],
                           item_rec_3, item_rec_4, W_item_self_34)
            # Time axis
            Con.connection(layers[layer_off+i*layer_space-4], layers[layer_off+i*layer_space+1],
                           item_rec_1,  item_rec_1, W_item_rec_11)
            Con.connection(layers[layer_off+i*layer_space-3], layers[layer_off+i*layer_space+2],
                           item_rec_2,  item_rec_2, W_item_rec_22)
            Con.connection(layers[layer_off+i*layer_space-2], layers[layer_off+i*layer_space+3],
                           item_rec_3,  item_rec_3, W_item_rec_33)
            Con.connection(layers[layer_off+i*layer_space-1], layers[layer_off+i*layer_space+4],
                           item_rec_4,  item_rec_4, W_item_rec_44)

            Con.connection(layers[layer_off+i*layer_space-4], layers[layer_off+i*layer_space+2],
                           item_rec_1,  item_rec_2, W_item_rec_12)
            Con.connection(layers[layer_off+i*layer_space-3], layers[layer_off+i*layer_space+3],
                           item_rec_2,  item_rec_3, W_item_rec_23)
            Con.connection(layers[layer_off+i*layer_space-2], layers[layer_off+i*layer_space+4],
                           item_rec_3,  item_rec_4, W_item_rec_34)

    #exit(0)        


def path_4rnn_pooling_4(pool_lay_name, pool_lay_dim, n_elements,
                        layers, layers_off, layers_space):
    pool_layer = Lay.PoolLayer(pool_lay_name, pool_lay_dim, "max")
    for idx in xrange(n_elements):
        Con.PoolConnection(layers[layers_off + idx*layers_space + layers_space-1], pool_layer)
    layers.append(pool_layer)

def path_4rnn_pooling_3(pool_lay_name, pool_lay_dim, n_elements,
                        layers, layers_off, layers_space):
    pool_layer = Lay.PoolLayer(pool_lay_name, pool_lay_dim, "max")
    for idx in xrange(n_elements):
        Con.PoolConnection(layers[layers_off + idx*layers_space + layers_space-2], pool_layer)
    layers.append(pool_layer)

def path_4rnn_pooling_2(pool_lay_name, pool_lay_dim, n_elements,
                        layers, layers_off, layers_space):
    pool_layer = Lay.PoolLayer(pool_lay_name, pool_lay_dim, "max")
    for idx in xrange(n_elements):
        Con.PoolConnection(layers[layers_off + idx*layers_space + layers_space-3], pool_layer)
    layers.append(pool_layer)

def path_4rnn_pooling_1(pool_lay_name, pool_lay_dim, n_elements,
                        layers, layers_off, layers_space):
    pool_layer = Lay.PoolLayer(pool_lay_name, pool_lay_dim, "max")
    for idx in xrange(n_elements):
        Con.PoolConnection(layers[layers_off + idx*layers_space + layers_space-4], pool_layer)
    layers.append(pool_layer)

def path_4rnn_pooling_0(pool_lay_name, pool_lay_dim, n_elements,
                        layers, layers_off, layers_space):
    pool_layer = Lay.PoolLayer(pool_lay_name, pool_lay_dim, "max")
    for idx in xrange(n_elements):
        Con.PoolConnection(layers[layers_off + idx*layers_space + layers_space-5], pool_layer)
    layers.append(pool_layer)

def SPT_RNN(SPT_words_l, SPT_Ws_l, SPT_GRs_l, SPT_POSs_l,
            SPT_words_r, SPT_Ws_r, SPT_GRs_r, SPT_POSs_r):
    """
    """
    n_words_l = len(SPT_words_l)
    n_words_r = len(SPT_words_r)
    n_words   = n_words_l + n_words_r

    n_Ws_l   = len(SPT_Ws_l)
    n_Ws_r   = len(SPT_Ws_r)
    n_Ws     = n_Ws_l + n_Ws_r

    n_GRs_l   = len(SPT_GRs_l)
    n_GRs_r   = len(SPT_GRs_r)
    n_GRs     = n_GRs_l + n_GRs_r

    layers = []

    ###### build rnns in two paths of words
    # build left path of words
    layer_space = 5
    #print 'SPT_RNN'
    #print len(B_emb_self)
    path_4rnn(B_emb_self, "embedding", vocab_dic, SPT_words_l,
             layers, 0, layer_space, emb_self,
             emb_rec_1, emb_rec_2, emb_rec_3, emb_rec_4,
             B_emb_rec_1, B_emb_rec_2, B_emb_rec_3, B_emb_rec_4,
             W_emb_self_01, W_emb_rec_11,
             W_emb_self_12, W_emb_rec_22, W_emb_rec_12,
             W_emb_self_23, W_emb_rec_33, W_emb_rec_23,
             W_emb_self_34, W_emb_rec_44, W_emb_rec_34)

    # build right path of words
    layer_words_r_off = len(layers)
    path_4rnn(B_emb_self, "embedding", vocab_dic, SPT_words_r,
             layers, layer_words_r_off, layer_space, emb_self,
             emb_rec_1, emb_rec_2, emb_rec_3, emb_rec_4,
             B_emb_rec_1, B_emb_rec_2, B_emb_rec_3, B_emb_rec_4,
             W_emb_self_01, W_emb_rec_11,
             W_emb_self_12, W_emb_rec_22, W_emb_rec_12,
             W_emb_self_23, W_emb_rec_33, W_emb_rec_23,
             W_emb_self_34, W_emb_rec_44, W_emb_rec_34)

    # build two pool layers of two paths of words
    layer_words_pool_off = len(layers)
    path_4rnn_pooling_0("pool_layer_words_l", emb_self,  n_words_l, layers, 0, layer_space)
    path_4rnn_pooling_1("pool_layer_words_l", emb_rec_1, n_words_l, layers, 0, layer_space)
    path_4rnn_pooling_2("pool_layer_words_l", emb_rec_2, n_words_l, layers, 0, layer_space)
    path_4rnn_pooling_3("pool_layer_words_l", emb_rec_3, n_words_l, layers, 0, layer_space)
    path_4rnn_pooling_4("pool_layer_words_l", emb_rec_4, n_words_l, layers, 0, layer_space)

    path_4rnn_pooling_0("pool_layer_words_r", emb_self,  n_words_r, layers, 0, layer_space)
    path_4rnn_pooling_1("pool_layer_words_r", emb_rec_1, n_words_r, layers, layer_words_r_off, layer_space)
    path_4rnn_pooling_2("pool_layer_words_r", emb_rec_2, n_words_r, layers, layer_words_r_off, layer_space)
    path_4rnn_pooling_3("pool_layer_words_r", emb_rec_3, n_words_r, layers, layer_words_r_off, layer_space)
    path_4rnn_pooling_4("pool_layer_words_r", emb_rec_4, n_words_r, layers, layer_words_r_off, layer_space)

    ###### build rnns in twos paths of POS
    # build left path of POS
    layer_pos_l_off = len(layers)
    path_4rnn(B_pos_self, "POS", pos_dic, SPT_POSs_l,
             layers, layer_pos_l_off, layer_space, pos_self,
             pos_rec_1, pos_rec_2, pos_rec_3, pos_rec_4,
             B_pos_rec_1, B_pos_rec_2, B_pos_rec_3, B_pos_rec_4,
             W_pos_self_01, W_pos_rec_11,
             W_pos_self_12, W_pos_rec_22, W_pos_rec_12,
             W_pos_self_23, W_pos_rec_33, W_pos_rec_23,
             W_pos_self_34, W_pos_rec_44, W_pos_rec_34)

    # build right path of POS
    layer_pos_r_off = len(layers)
    path_4rnn(B_pos_self, "POS", pos_dic, SPT_POSs_r,
             layers, layer_pos_r_off, layer_space, pos_self,
             pos_rec_1, pos_rec_2, pos_rec_3, pos_rec_4,
             B_pos_rec_1, B_pos_rec_2, B_pos_rec_3, B_pos_rec_4,
             W_pos_self_01, W_pos_rec_11,
             W_pos_self_12, W_pos_rec_22, W_pos_rec_12,
             W_pos_self_23, W_pos_rec_33, W_pos_rec_23,
             W_pos_self_34, W_pos_rec_44, W_pos_rec_34)

    # build two pool layers of two paths of POS
    layer_pos_pool_off = len(layers)
    path_4rnn_pooling_0("pool_layer_POS_l", pos_self,  n_words_l, layers, layer_pos_l_off, layer_space)
    path_4rnn_pooling_1("pool_layer_POS_l", pos_rec_1, n_words_l, layers, layer_pos_l_off, layer_space)
    path_4rnn_pooling_2("pool_layer_POS_l", pos_rec_2, n_words_l, layers, layer_pos_l_off, layer_space)
    path_4rnn_pooling_3("pool_layer_POS_l", pos_rec_3, n_words_l, layers, layer_pos_l_off, layer_space)
    path_4rnn_pooling_4("pool_layer_POS_l", pos_rec_4, n_words_l, layers, layer_pos_l_off, layer_space)

    path_4rnn_pooling_0("pool_layer_POS_r", pos_self,  n_words_r, layers, layer_pos_r_off, layer_space)
    path_4rnn_pooling_1("pool_layer_POS_r", pos_rec_1, n_words_r, layers, layer_pos_r_off, layer_space)
    path_4rnn_pooling_2("pool_layer_POS_r", pos_rec_2, n_words_r, layers, layer_pos_r_off, layer_space)
    path_4rnn_pooling_3("pool_layer_POS_r", pos_rec_3, n_words_r, layers, layer_pos_r_off, layer_space)
    path_4rnn_pooling_4("pool_layer_POS_r", pos_rec_4, n_words_r, layers, layer_pos_r_off, layer_space)

    ###### build rnns in twos paths of WN
    # build left path of WN
    layer_wn_l_off = len(layers)
    path_4rnn(B_wn_self, "WordNet", wn_dic, SPT_Ws_l,
             layers, layer_wn_l_off, layer_space, wn_self,
             wn_rec_1, wn_rec_2, wn_rec_3, wn_rec_4,
             B_wn_rec_1, B_wn_rec_2, B_wn_rec_3, B_wn_rec_4,
             W_wn_self_01, W_wn_rec_11,
             W_wn_self_12, W_wn_rec_22, W_wn_rec_12,
             W_wn_self_23, W_wn_rec_33, W_wn_rec_23,
             W_wn_self_34, W_wn_rec_44, W_wn_rec_34)

    # build right path of WN
    layer_wn_r_off = len(layers)
    path_4rnn(B_wn_self, "WordNet", wn_dic, SPT_Ws_r,
             layers, layer_wn_r_off, layer_space, wn_self,
             wn_rec_1, wn_rec_2, wn_rec_3, wn_rec_4,
             B_wn_rec_1, B_wn_rec_2, B_wn_rec_3, B_wn_rec_4,
             W_wn_self_01, W_wn_rec_11,
             W_wn_self_12, W_wn_rec_22, W_wn_rec_12,
             W_wn_self_23, W_wn_rec_33, W_wn_rec_23,
             W_wn_self_34, W_wn_rec_44, W_wn_rec_34)

    # build two pool layers of two paths of WN
    layer_wn_pool_off = len(layers)

    path_4rnn_pooling_0("pool_layer_WN_l", wn_self,  n_Ws_l, layers, layer_wn_l_off, layer_space)
    path_4rnn_pooling_1("pool_layer_WN_l", wn_rec_1, n_Ws_l, layers, layer_wn_l_off, layer_space)
    path_4rnn_pooling_2("pool_layer_WN_l", wn_rec_2, n_Ws_l, layers, layer_wn_l_off, layer_space)
    path_4rnn_pooling_3("pool_layer_WN_l", wn_rec_3, n_Ws_l, layers, layer_wn_l_off, layer_space)
    path_4rnn_pooling_4("pool_layer_WN_l", wn_rec_3, n_Ws_l, layers, layer_wn_l_off, layer_space)

    path_4rnn_pooling_0("pool_layer_WN_r", wn_self,  n_Ws_r, layers, layer_wn_r_off, layer_space)
    path_4rnn_pooling_1("pool_layer_WN_r", wn_rec_1, n_Ws_r, layers, layer_wn_r_off, layer_space)
    path_4rnn_pooling_2("pool_layer_WN_r", wn_rec_2, n_Ws_r, layers, layer_wn_r_off, layer_space)
    path_4rnn_pooling_3("pool_layer_WN_r", wn_rec_3, n_Ws_r, layers, layer_wn_r_off, layer_space)
    path_4rnn_pooling_4("pool_layer_WN_r", wn_rec_3, n_Ws_r, layers, layer_wn_r_off, layer_space)

    ###### build rnns in twos paths of GR
    # build left path of GR
    layer_gr_l_off = len(layers)
    path_4rnn(B_gr_self, "GR", gr_dic, SPT_GRs_l,
             layers, layer_gr_l_off, layer_space, gr_self,
             gr_rec_1, gr_rec_2, gr_rec_3, gr_rec_4,
             B_gr_rec_1, B_gr_rec_2, B_gr_rec_3, B_gr_rec_4,
             W_gr_self_01, W_gr_rec_11,
             W_gr_self_12, W_gr_rec_22, W_gr_rec_12,
             W_gr_self_23, W_gr_rec_33, W_gr_rec_23,
             W_gr_self_34, W_gr_rec_44, W_gr_rec_34)

    # build right path of GR
    layer_gr_r_off = len(layers)
    path_4rnn(B_gr_self, "GR", gr_dic, SPT_GRs_r,
             layers, layer_gr_r_off, layer_space, gr_self,
             gr_rec_1, gr_rec_2, gr_rec_3, gr_rec_4,
             B_gr_rec_1, B_gr_rec_2, B_gr_rec_3, B_gr_rec_4,
             W_gr_self_01, W_gr_rec_11,
             W_gr_self_12, W_gr_rec_22, W_gr_rec_12,
             W_gr_self_23, W_gr_rec_33, W_gr_rec_23,
             W_gr_self_34, W_gr_rec_44, W_gr_rec_34)

    # build two pool layers of two paths of GR
    layer_gr_pool_off = len(layers)
    path_4rnn_pooling_0("pool_layer_GR_l", gr_self,  n_GRs_l, layers, layer_gr_l_off, layer_space)
    path_4rnn_pooling_1("pool_layer_GR_l", gr_rec_1, n_GRs_l, layers, layer_gr_l_off, layer_space)
    path_4rnn_pooling_2("pool_layer_GR_l", gr_rec_2, n_GRs_l, layers, layer_gr_l_off, layer_space)
    path_4rnn_pooling_3("pool_layer_GR_l", gr_rec_3, n_GRs_l, layers, layer_gr_l_off, layer_space)
    path_4rnn_pooling_4("pool_layer_GR_l", gr_rec_4, n_GRs_l, layers, layer_gr_l_off, layer_space)

    path_4rnn_pooling_0("pool_layer_GR_r", gr_self,  n_GRs_r, layers, layer_gr_r_off, layer_space)
    path_4rnn_pooling_1("pool_layer_GR_r", gr_rec_1, n_GRs_r, layers, layer_gr_r_off, layer_space)
    path_4rnn_pooling_2("pool_layer_GR_r", gr_rec_2, n_GRs_r, layers, layer_gr_r_off, layer_space)
    path_4rnn_pooling_3("pool_layer_GR_r", gr_rec_3, n_GRs_r, layers, layer_gr_r_off, layer_space)
    path_4rnn_pooling_4("pool_layer_GR_r", gr_rec_3, n_GRs_r, layers, layer_gr_r_off, layer_space)

    ##### build last hidden layer
    layer_hid_off = len(layers)
    hid_layer = Lay.layer("last_hidden", B_hid, num_hid)
    hid_layer.act = "hidden"
    layers.append(hid_layer)

    Con.connection(layers[layer_words_pool_off],   hid_layer, emb_self,  num_hid, W_emb_hid_0[ : emb_self*num_hid])
    Con.connection(layers[layer_words_pool_off+1], hid_layer, emb_rec_1, num_hid, W_emb_hid_1[ : emb_rec_1*num_hid])
    Con.connection(layers[layer_words_pool_off+2], hid_layer, emb_rec_2, num_hid, W_emb_hid_2[ : emb_rec_2*num_hid])
    Con.connection(layers[layer_words_pool_off+3], hid_layer, emb_rec_3, num_hid, W_emb_hid_3[ : emb_rec_3*num_hid])
    Con.connection(layers[layer_words_pool_off+4], hid_layer, emb_rec_4, num_hid, W_emb_hid_4[ : emb_rec_4*num_hid])

    Con.connection(layers[layer_words_pool_off+5], hid_layer, emb_self,  num_hid, W_emb_hid_0[emb_self*num_hid : ])
    Con.connection(layers[layer_words_pool_off+6], hid_layer, emb_rec_1, num_hid, W_emb_hid_1[emb_rec_1*num_hid : ])
    Con.connection(layers[layer_words_pool_off+7], hid_layer, emb_rec_2, num_hid, W_emb_hid_2[emb_rec_2*num_hid : ])
    Con.connection(layers[layer_words_pool_off+8], hid_layer, emb_rec_3, num_hid, W_emb_hid_3[emb_rec_3*num_hid : ])
    Con.connection(layers[layer_words_pool_off+9], hid_layer, emb_rec_4, num_hid, W_emb_hid_4[emb_rec_4*num_hid : ])
    #########################################################
    Con.connection(layers[layer_pos_pool_off],     hid_layer, pos_self, num_hid,  W_pos_hid_0[ : pos_self*num_hid])
    Con.connection(layers[layer_pos_pool_off+1],   hid_layer, pos_rec_1, num_hid, W_pos_hid_1[ : pos_rec_1*num_hid])
    Con.connection(layers[layer_pos_pool_off+2],   hid_layer, pos_rec_2, num_hid, W_pos_hid_2[ : pos_rec_2*num_hid])
    Con.connection(layers[layer_pos_pool_off+3],   hid_layer, pos_rec_3, num_hid, W_pos_hid_3[ : pos_rec_3*num_hid])
    Con.connection(layers[layer_pos_pool_off+4],   hid_layer, pos_rec_4, num_hid, W_pos_hid_4[ : pos_rec_4*num_hid])

    Con.connection(layers[layer_pos_pool_off+5],   hid_layer, pos_self, num_hid,  W_pos_hid_0[pos_self*num_hid : ])
    Con.connection(layers[layer_pos_pool_off+6],   hid_layer, pos_rec_1, num_hid, W_pos_hid_1[num_hid*pos_rec_1 : ])
    Con.connection(layers[layer_pos_pool_off+7],   hid_layer, pos_rec_2, num_hid, W_pos_hid_2[num_hid*pos_rec_2 : ])
    Con.connection(layers[layer_pos_pool_off+8],   hid_layer, pos_rec_3, num_hid, W_pos_hid_3[num_hid*pos_rec_3 : ])
    Con.connection(layers[layer_pos_pool_off+9],   hid_layer, pos_rec_4, num_hid, W_pos_hid_4[num_hid*pos_rec_4 : ])
    #########################################################
    Con.connection(layers[layer_wn_pool_off],      hid_layer, wn_self,   num_hid, W_wn_hid_0[ : wn_self*num_hid])
    Con.connection(layers[layer_wn_pool_off+1],    hid_layer, wn_rec_1,  num_hid, W_wn_hid_1[ : wn_rec_1*num_hid])
    Con.connection(layers[layer_wn_pool_off+2],    hid_layer, wn_rec_2,  num_hid, W_wn_hid_2[ : wn_rec_2*num_hid])
    Con.connection(layers[layer_wn_pool_off+3],    hid_layer, wn_rec_3,  num_hid, W_wn_hid_3[ : wn_rec_3*num_hid])
    Con.connection(layers[layer_wn_pool_off+4],    hid_layer, wn_rec_4,  num_hid, W_wn_hid_4[ : wn_rec_4*num_hid])

    Con.connection(layers[layer_wn_pool_off+5],    hid_layer, wn_self,   num_hid, W_wn_hid_0[wn_self*num_hid : ])
    Con.connection(layers[layer_wn_pool_off+6],    hid_layer, wn_rec_1,  num_hid, W_wn_hid_1[num_hid*wn_rec_1 : ])
    Con.connection(layers[layer_wn_pool_off+7],    hid_layer, wn_rec_2,  num_hid, W_wn_hid_2[num_hid*wn_rec_2 : ])
    Con.connection(layers[layer_wn_pool_off+8],    hid_layer, wn_rec_3,  num_hid, W_wn_hid_3[num_hid*wn_rec_3 : ])
    Con.connection(layers[layer_wn_pool_off+9],    hid_layer, wn_rec_4,  num_hid, W_wn_hid_4[num_hid*wn_rec_4 : ])
    #########################################################
    Con.connection(layers[layer_gr_pool_off],      hid_layer, gr_self,   num_hid, W_gr_hid_0[ : gr_self*num_hid])
    Con.connection(layers[layer_gr_pool_off+1],    hid_layer, gr_rec_1,  num_hid, W_gr_hid_1[ : gr_rec_1*num_hid])
    Con.connection(layers[layer_gr_pool_off+2],    hid_layer, gr_rec_2,  num_hid, W_gr_hid_2[ : gr_rec_2*num_hid])
    Con.connection(layers[layer_gr_pool_off+3],    hid_layer, gr_rec_3,  num_hid, W_gr_hid_3[ : gr_rec_3*num_hid])
    Con.connection(layers[layer_gr_pool_off+4],    hid_layer, gr_rec_4,  num_hid, W_gr_hid_4[ : gr_rec_4*num_hid])

    Con.connection(layers[layer_gr_pool_off+5],    hid_layer, gr_self,   num_hid, W_gr_hid_0[gr_self*num_hid : ])
    Con.connection(layers[layer_gr_pool_off+6],    hid_layer, gr_rec_1,  num_hid, W_gr_hid_1[num_hid*gr_rec_1 : ])
    Con.connection(layers[layer_gr_pool_off+7],    hid_layer, gr_rec_2,  num_hid, W_gr_hid_2[num_hid*gr_rec_2 : ])
    Con.connection(layers[layer_gr_pool_off+8],    hid_layer, gr_rec_3,  num_hid, W_gr_hid_3[num_hid*gr_rec_3 : ])
    Con.connection(layers[layer_gr_pool_off+9],    hid_layer, gr_rec_4,  num_hid, W_gr_hid_4[num_hid*gr_rec_4 : ])

    ##### build output layer
    out_layer = Lay.layer("output", B_out, num_out)
    if gl.numOut > 0:
        out_layer._activate = Activation.softmax
        out_layer._activatePrime = None
        out_layer.act = "softmax"
    else:
        out_layer._activate = Activation.sigmoid
        out_layer._activatePrime = Activation.sigmoidPrime
    Con.connection(layers[layer_hid_off], out_layer, num_hid, num_out, W_hid_out[ : num_hid*num_out])
    layers.append(out_layer)

    return layers

def write_one_net(layers, fname):
    """
    """
    f = file(fname, 'wb')
    num_lay = struct.pack('i', len(layers))
    f.write(num_lay)

    #################################
    # pre-processing, compute some indexes
    num_con = 0
    for i, layer in enumerate(layers):
        layer.idx = i
        num_con += len(layer.connectUp)
        for (icon, con) in enumerate(layer.connectDown):
            con.ydownid = icon

    num_con = struct.pack('i', num_con)
    f.write(num_con)

    ###### layers ######
    for layer in layers:
        # name
        # = struct.pack('s', layer.name )
        # numUnit
        tmp = struct.pack('i', layer.numUnit )
        f.write( tmp )
        # numUp
        tmp = struct.pack('i', len(layer.connectUp))
        f.write( tmp )
        # numDown
        tmp = struct.pack('i', len(layer.connectDown))
        f.write( tmp )

        if  layer.layer_type == 'p': # pooling
            if layer.poolType == 'max':
                tlayer = 'x'
            elif layer.poolType == 'sum':
                tlayer = 'u'
            tmp = struct.pack('c', tlayer)
            f.write(tmp)
        elif layer.layer_type == 'o': # ordinary nodes
            if layer.act == 'embedding':
                tlayer = 'e'
            elif layer.act == 'POS':
                tlayer = 'p'
            elif layer.act == 'GR':
                tlayer = 'm'
            elif layer.act == "WordNet":
                tlayer = 'w'
            elif layer.act == "NER":
                tlayer = 'n'
            elif layer.act == "ReLU":
                tlayer = 'r'
            elif layer.act == 'tanh':
                tlayer = 't'
            elif layer.act == "sigmoid":
                tlayer = 'g'
            elif layer.act == 'softmax':
                tlayer = 's'
            elif layer.act == "dummy":
                tlayer = 'd'
            elif layer.act == "hidden":
                tlayer = 'h'
            else:
                print "error:", layer.act, layer.name
                return
            tmp = struct.pack('c', tlayer)

            f.write( tmp )
            bidx = -1
            if layer.bidx != -1:
                bidx = layer.bidx[0]
            tmp = struct.pack('i', bidx)
            f.write(tmp)

    ###### connections ######
    for layer in layers:
#         print layer.name
        for xupid, con in enumerate(layer.connectUp):
            # connection type
            tmp = struct.pack('c', con.con_type)
            f.write(tmp)

            # xlayer idx
            if(con.con_type == 'b'):
                tmp = struct.pack('i', con.xlayer1.idx)
            else:
                tmp = struct.pack('i', con.xlayer.idx)
            f.write( tmp )

            # ylayer idx
            tmp = struct.pack('i', con.ylayer.idx )
            f.write( tmp)

            # idx in x's connectUp
            tmp = struct.pack('i', xupid)
            f.write( tmp )

            # idx in y's connectDown
            tmp = struct.pack('i', con.ydownid)
            f.write( tmp )

            if con.con_type == 'b': # Widx and xlayer2 for bilinear connection
                tmp = struct.pack('i', con.Widx)
                f.write(tmp)
                if con.xlayer2 != None:
                    x2 = con.xlayer2.idx
                else:
                    x2 = -1
                tmp = struct.pack('i', x2)
                f.write(tmp)
            elif con.con_type == 'p': # pooling connection
                Widx = -1
                tmp = struct.pack('i', Widx)
                f.write(tmp)
            else: # ordinary connection
                if con.Widx == -1:
                    Widx == -1
                else:
                    Widx = con.Widx[0]
                tmp = struct.pack('i', Widx)
                f.write(tmp)
                if Widx >= 0:
                    tmp = struct.pack('f', con.Wcoef)
                    f.write(tmp)
    f.close()
'''

'''

def build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e):
    """
    """
    SPTs_train = load_rev_valid_aug18class_samples_train(path_raw_data, file_train)
    SPTs_test  = load_rev_direc_samples_test(path_raw_data, file_test)
    print('build_SDPs_RNNs')
    #print len(SPTs_train)
    #print SPTs_train[0]
    # Training Data Loaded
    #print train_b, train_e
    for i in xrange(train_b, train_e):
        SPT = SPTs_train[i]
        #print i 
        #print 'SPT'
        #print SPT
        #print '\n'
        #exit(0)
        #print SPT[4], SPT[10], SPT[8], SPT[6],SPT[5], SPT[11], SPT[9], SPT[7]
        #exit(0)
        layers = SPT_RNN(SPT[4], SPT[10], SPT[8], SPT[6],
                         SPT[5], SPT[11], SPT[9], SPT[7])

        path_file_name = path_train + str(i) + ".net"
        write_one_net(layers, path_file_name)
        #print i

    for j in xrange(valid_b, valid_e):
        SPT = SPTs_train[j]
        layers = SPT_RNN(SPT[4], SPT[10], SPT[8], SPT[6],
                         SPT[5], SPT[11], SPT[9], SPT[7])
        path_file_name = path_valid + str(j) + ".net"
        write_one_net(layers, path_file_name)
        #print j

    for k in xrange(test_b, test_e):
        SPT = SPTs_test[k]
        layers = SPT_RNN(SPT[4], SPT[10], SPT[8], SPT[6],
                         SPT[5], SPT[11], SPT[9], SPT[7])
        path_file_name = path_test + str(k) + ".net"
        write_one_net(layers, path_file_name)
        #print k

def load_SPT_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e):
    """
    """
    file_train = "sem_train_8000.txt"
    file_test  = "sem_test_2717.txt"

    X_train = []; Y_train = []
    X_valid = []; Y_valid = []
    X_test  = []; Y_test  = []

    # Loading the labels here. Whats the need to pass the file_train

    labels_train = load_rev_valid_aug18class_labels_train(path_raw_data, file_train)
    labels_test  = load_rev_direc_labels_test(path_raw_data, file_test)


    ### train ###
    
    for i in xrange(train_b, train_e):
        path_file_name = path_train + str(i) + ".net"
        if os.path.exists(path_file_name):
            X_train.append(path_file_name)
            Y_train.append(labels_train[i])

    ### valid ###
    for j in xrange(valid_b, valid_e):
        path_file_name = path_valid + str(j) + ".net"
        if os.path.exists(path_file_name):
            X_valid.append(path_file_name)
            Y_valid.append(labels_train[j])

    ### test ###
    for k in xrange(test_b, test_e):
        path_file_name = path_test + str(k) + ".net"
        if os.path.exists(path_file_name):
            X_test.append(path_file_name)
            Y_test.append(labels_test[k])

    return (X_train, train_b, train_e), (X_valid, valid_b, valid_e), (X_test, test_b, test_e), \
           (Y_train, train_b, train_e), (Y_valid, valid_b, valid_e), (Y_test, test_b, test_e)

def shuffle_SPT_RNNs(X_train, Y_train, seed_tr):
    """
    """
    np.random.seed(seed_tr)
    np.random.shuffle(X_train)
    np.random.seed(seed_tr)
    np.random.shuffle(Y_train)

def save_SPT_RNNs_in_one_file(Tuple_X, out_file_name):
    """
    """
    #print 'writing ', out_file_name
    X, idx_b, idx_e = Tuple_X
    fout = file(path_join+out_file_name+"_"+str(idx_b)+"_"+str(idx_e)+".nets", 'wb')

    for num, x in enumerate(X):
        if num % 1000 == 0:
            print num

        fin = file(str(X[num]), 'rb')
        tmpstr = fin.read()
        if len(tmpstr) <= 10:
            print "great error! num = ", num, "; file = ", X[num]
        fout.write(tmpstr)
        fin.close()
    fout.close()

def save_labels_in_one_file(file_name, Tuple_Y_train, Tuple_Y_valid, Tuple_Y_test):
    """
    """

    Y_train, train_b, train_e = Tuple_Y_train
    Y_valid, valid_b, valid_e = Tuple_Y_valid
    Y_test,  test_b,  test_e  = Tuple_Y_test

    f = file(path_join+file_name+"_"+str(train_b)+"_"+str(train_e) \
                                +"_"+str(valid_b)+"_"+str(valid_e) \
                                +"_"+str(test_b) +"_"+str(test_e)+".labels", 'w')

    f.write(str(len(Y_train)) + '\n')
    f.write(str(len(Y_valid)) + '\n')
    f.write(str(len(Y_test))  + '\n')

    for i in Y_train:
        f.write(str(i) + '\n')
    for i in Y_valid:
        f.write(str(i) + '\n')
    for i in Y_test:
        f.write(str(i) + '\n')

    f.close()


def put_dataset_in_one_file(Tuple_X_train, Tuple_X_valid, Tuple_X_test,
                            Tuple_Y_train, Tuple_Y_valid, Tuple_Y_test):
    """
    """
    save_SPT_RNNs_in_one_file(Tuple_X_train, 'train')
    save_SPT_RNNs_in_one_file(Tuple_X_valid, 'valid')
    save_SPT_RNNs_in_one_file(Tuple_X_test,  'test')
    save_labels_in_one_file("label", Tuple_Y_train, Tuple_Y_valid, Tuple_Y_test)

def save_paras(train_b, train_e, valid_b, valid_e, test_b, test_e, file_name, W, B):
    """
    """
    f = file(path_join+file_name+"_"+str(train_b)+"_"+str(train_e) \
                                +"_"+str(valid_b)+"_"+str(valid_e) \
                                +"_"+str(test_b) +"_"+str(test_e)+".paras", 'wb')
    print path_join+file_name+"_"+str(train_b)+"_"+str(train_e) \
                                +"_"+str(valid_b)+"_"+str(valid_e) \
                                +"_"+str(test_b) +"_"+str(test_e)+".paras"

    numW = struct.pack('i', len(W))
    numB = struct.pack('i', len(B))
    f.write(numW)
    f.write(numB)

    print "numW:", len(W)
    print "numB:", len(B)

    for i in xrange(len(W)):
        tmp = struct.pack('f', W[i])
        f.write(tmp)

    for i in xrange(len(B)):
        tmp = struct.pack('f', B[i])
        f.write(tmp)

    f.close()

def save_shuffled_files_order(path, data_type, Tuple_data):
    """
    """
    path_file_names = Tuple_data[0]
    idx_b = Tuple_data[1]
    idx_e = Tuple_data[2]
    file_name = data_type + \
                "_Beg" + str(idx_b) + "_End" + str(idx_e) + ".txt";
    print "file_name:", file_name

    fout = open(path + file_name, "w")

    for idx, file in enumerate(path_file_names):
        file_order = file[file.rindex("/")+1 : file.rindex(".")]
#         print idx, int(file_order)+1
        fout.write(str(int(file_order)+1) + "\n")

    fout.close()

def load_shuffled_files_order(path, data_type, seed, idx_b, idx_e):
    """
    """
    file_name = data_type + "_Seed" + str(seed) + \
                "_Beg" + str(idx_b) + "_End" + str(idx_e) + ".txt";

    file_orders = []
    fout = open(path + file_name, "r");
    for line in fout:
        file_orders.append(line)
    fout.close()

    return file_orders;

def load_shuffled_labels(path, data_type, seed, idx_b, idx_e):
    """
    """
    file_name = data_type + "_" + \
                str(seed) + "_" + \
                str(idx_b) + "_" + \
                str(idx_e) + "_labels.txt"
    labels = []
    fout = open(path + file_name, "r");
    for line in fout:
        labels.append(line)
    fout.close()

    return labels

def save_proposed_answer_txt(path, data_type, seed, idx_b, idx_e, accuracy):
    """
    """
    file_orders = load_shuffled_files_order(path, data_type, seed, idx_b, idx_e)
    labels = load_shuffled_labels(path, data_type, seed, idx_b, idx_e)

    fout = open(path + data_type + "_proposed_answer_" + str(accuracy) + ".txt", "w")

    for idx, ele in enumerate(file_orders):
        print ele.strip(), "\t", labels[idx].strip()
        fout.write(ele.strip() + "\t" + labels[idx])

    fout.close()

def check_built_nn(layers):
    print 'Totally', len(layers), 'layer(s)'
    for l in layers:
        print l.name, l.layer_type,
        if l.layer_type != 'p':
            print l.act
        for con in l.connectDown:
            print '   ', con.con_type,
            if con.con_type == 'b':
                if con.xlayer2 == None:
                    x2name = 'none'
                else:
                    x2name = con.xlayer2.name
                    print '(', con.xlayer1.name, x2name, ')->', con.ylayer.name
            else:
                print con.xlayer.name, '->', con.ylayer.name


#--------------------------------------------------------------------#
if __name__ == '__main__':
    """
    """
    print "hi there"
    train_b = 800;
    train_e = 13913
    valid_b = 0;
    valid_e = 800
    test_b  = 0;
    test_e  = 2717

    seed_train = 1000

    build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)

    Tuple_X_train, Tuple_X_valid, Tuple_X_test, Tuple_Y_train, Tuple_Y_valid, Tuple_Y_test = \
        load_SPT_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)

    print 'Here is training data'
    
    
    #print Tuple_X_train.   ...one file is one training data 
    #print Tuple_Y_train
    #exit(0)    

    shuffle_SPT_RNNs(Tuple_X_train[0], Tuple_Y_train[0], seed_train)

    save_shuffled_files_order(path_join, "Test", Tuple_X_test)

    put_dataset_in_one_file(Tuple_X_train, Tuple_X_valid, Tuple_X_test,
                            Tuple_Y_train, Tuple_Y_valid, Tuple_Y_test)

    save_paras(train_b, train_e, valid_b, valid_e, test_b, test_e, "paras", Weights, Biases)
