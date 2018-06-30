# coding=utf-8
'''
@author: ***  Chandan Pandey
So My first model will be a simple LSTM classifier for relations.  

'''
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix 
# To import All the relavant files :D 
sys.path.append('../data/')
sys.path.append('../util/')
sys.path.append('../Nets/')
sys.path.append('../nn/')
from tensorflow.contrib import learn


from SemEval2010 import load_rev_direc_samples_test, load_rev_direc_labels_test,\
    load_rev_valid_aug18class_samples_train, load_rev_valid_aug18class_labels_train, \
    load_wrd_vec_dic_v2, lst_2_dic, load_emd_lst_v2, WordNet_44_categories, \
    POS_15_categories, GR_19_categories
from file_io import get_curr_file_name_no_pfix
import os
import numpy as np
import cPickle  # @UnusedImport
import struct

##############################
# hyper_param
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
gr_rec_1 = 25
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
path_valid = dir1 + dir2 + dir3 + "/valid/"
path_test  = dir1 + dir2 + dir3 + "/test/"
path_join  = dir1 + dir2 + dir3 + "/join/"

    
learning_rate = 0.001
file_train = "sem_train_8000.txt"
file_test = "sem_test_2717.txt"

vocab_dic  = load_wrd_vec_dic_v2(path_raw_data)

vocab_dic['paddd'] = [len(vocab_dic), np.zeros(200)] 

emb_lst    = load_emd_lst_v2(path_raw_data)
emb_lst.append(np.zeros(200))
Embeddings = np.array(emb_lst)

wn_num  = 10
pos_num = 15
gr_num  = 19
wn_dic  = lst_2_dic(WordNet_44_categories)
pos_dic = lst_2_dic(POS_15_categories)
gr_dic  = lst_2_dic(GR_19_categories)


#wn_embeddings = np.zeros(shape=(45, 50))
#pos_embeddings = np.zeros(shape=(16, 50))
gr_embeddings = np.random.uniform(-0.05, 0.05, (20, 25))

wn_embeddings = np.random.uniform(-0.05, 0.05, (45, 25))
pos_embeddings = np.random.uniform(-0.05, 0.05, (16, 25))


wn_dic['paddd'] = 0
pos_dic['paddd'] = 0
gr_dic['paddd'] = 0

##############################
'''
Model: LSTM with first  layer of size 128  
'''

num_epochs = 120
truncated_backprop_length = 12
state_size = 150
num_classes = 19
batch_size = 50  # Original 50 gave 84.35 F-1 without validation, 60 size does not help much

SGD_lrate1 = [0.55, 0.54, 0.536, 0.53, 0.525, 0.521, 0.53, 0.516, 0.51, 0.50, 0.48, 0.477, 0.460, 0.445, 0.433, 0.428, 0.420, 0.39, 0.36, 0.348, 0.32, 0.31, 
                0.3, 0.287, 0.27, 0.243, 0.00443426, 0.0039, 0.00343426, 0.00253426, 0.00213426, 0.00194, 0.00188, 0.00186, 0.00178, 0.00160, 0.00151, 
                0.0014, 0.0013, 0.0011, 0.001, 0.0009, 0.2187, 0.19683, 0.177147, 0.3, 0.27, 0.268, 0.265, 0.26, 0.22, 0.21, 0.17,
                0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 
                0.0500315, 0.0450284, 0.0405255, 0.036473, 0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 
                0.0174449, 0.0157004, 0.0141304, 0.0127173, 0.0114456, 0.010301, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0039, 0.00343426, 0.00253426, 0.00213426, 0.00194, 0.00188, 0.00186, 0.00178, 0.00160, 0.00151, 
                0.0014, 0.0013, 0.0011, 0.001, 0.0009
                ]

SGD_lrate2 = [0.812,0.795,0.775,0.754, 0.732, 0.72, 0.714, 0.694, 0.673,0.6543,0.63,0.612,0.593,0.57,0.55, 0.54, 0.536, 0.53, 0.525, 0.521, 0.53, 0.516, 0.51, 0.50, 0.48, 0.477, 0.460, 0.445, 0.433, 0.428, 0.420, 0.39, 0.36, 0.348, 0.32, 0.31, 
                0.3, 0.287, 0.27, 0.243, 0.2187, 0.19683, 0.177147, 0.3, 0.27, 0.268, 0.265, 0.26, 0.22, 0.21, 0.17,
                0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 
                0.0500315, 0.0450284, 0.0405255, 0.036473, 0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 
                0.0174449, 0.0157004, 0.0141304, 0.0127173, 0.0114456, 0.010301, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0039, 0.00343426, 0.00253426, 0.00213426, 0.00194, 0.00188, 0.00186, 0.00178, 0.00160, 0.00151, 
                0.0014, 0.0013, 0.0011, 0.001, 0.0009
                ]


SGD_lrate = [0.39, 0.36, 0.348, 0.32, 0.31, 0.3, 0.287, 0.27, 0.243, 0.2187, 0.19683, 0.177147, 0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 
                0.0500315, 0.0450284, 0.0405255, 0.036473, 0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 
                0.0174449, 0.0157004, 0.0141304, 0.0127173, 0.0114456, 0.010301, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0039, 0.00343426, 0.00253426]



SGD_lrate3 = [0.812,0.795,0.775,0.754, 0.732, 0.72, 0.714, 0.694, 0.673,0.6543,0.63,0.612,0.593,0.57,
                0.55, 0.54, 0.536, 0.53, 0.525, 0.521, 0.53, 0.516, 0.51, 0.50, 0.48, 0.477, 0.460, 0.445, 0.433, 0.428, 0.420, 0.39, 0.36, 0.348, 0.32, 0.31, 
                0.3, 0.287, 0.27, 0.243, 0.2187, 0.19683, 0.177147, 0.3, 0.27, 0.268, 0.265, 0.26, 0.22, 0.21, 0.17,
                0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 
                0.0500315, 0.0450284, 0.0405255, 0.036473, 0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 
                0.0174449, 0.0157004, 0.0141304, 0.0127173, 0.0114456, 0.010301, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0039, 0.00343426, 0.00253426, 0.00213426, 0.00194, 0.00188, 0.00186, 0.00178, 0.00160, 0.00151, 
                0.0014, 0.0013, 0.0011, 0.001, 0.0009
                
                ]


SGD_lrate4 = [0.812,0.795,0.775,0.754, 0.732, 0.72, 0.714, 0.694, 0.673,0.6543,0.63,0.612,0.593,0.57,
                0.55, 0.54, 0.536, 0.53, 0.525, 0.521, 0.53, 0.516, 0.51, 0.50, 0.48, 0.477, 0.460, 0.445, 0.433, 0.428, 0.420, 0.39, 0.36, 0.348, 0.32, 0.31, 
                0.3, 0.287, 0.27, 0.243, 0.00443426, 0.0039, 0.00343426, 0.00253426, 0.00213426, 0.00194, 0.00188, 0.00186, 0.00178, 0.00160, 0.00151, 
                0.0014, 0.0013, 0.0011, 0.001, 0.0009, 0.2187, 0.19683, 0.177147, 0.3, 0.27, 0.268, 0.265, 0.26, 0.22, 0.21, 0.17,
                0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 
                0.0500315, 0.0450284, 0.0405255, 0.036473, 0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 
                0.0174449, 0.0157004, 0.0141304, 0.0127173, 0.0114456, 0.010301, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0039, 0.00343426, 0.00253426, 0.00213426, 0.00194, 0.00188, 0.00186, 0.00178, 0.00160, 0.00151, 
                0.0014, 0.0013, 0.0011, 0.001, 0.0009
                ]


SGD_lrate5 = [0.27, 0.287, 0.3, 0.31, 0.32, 0.348, 0.36, 0.39, 0.42, 0.428, 0.433, 0.445, 0.46, 0.477, 0.48, 0.5, 0.51, 0.516, 0.53, 0.521, 0.525, 0.53, 0.536, 
                0.54, 0.55, 0.57, 0.593, 0.612, 0.63, 0.6543, 0.673, 0.694, 0.714, 0.72, 0.732, 0.754, 0.775, 0.795, 0.812, 0.82, 0.825, 0.83, 0.837, 0.84, 
                0.839, 0.835, 0.836, 0.833, 0.829, 0.825, 0.82, 0.816, 0.810, 0.805, 0.8, 
                0.774, 0.77, 0.766, 0.762, 0.758, 0.754, 0.75, 0.746, 0.742, 0.738, 0.734, 0.73, 0.726, 0.722, 0.718, 0.714, 0.71, 0.706, 0.702, 0.698, 
                0.694, 0.69, 0.6859999999999999, 0.6819999999999999, 0.6779999999999999, 0.6739999999999999, 0.6699999999999999, 0.6659999999999999, 
                0.6619999999999999, 0.6579999999999999, 0.6539999999999999, 0.6499999999999999, 0.6459999999999999, 0.6419999999999999, 0.6379999999999999, 
                0.6339999999999999, 0.6299999999999999, 0.6259999999999999, 0.6219999999999999, 0.6179999999999999, 0.6139999999999999, 0.6099999999999999, 
                0.6059999999999999, 0.6019999999999999, 0.5979999999999999, 0.5939999999999999, 0.5899999999999999, 0.5859999999999999, 0.5819999999999999, 
                0.5779999999999998, 0.5739999999999998, 0.5699999999999998, 0.5659999999999998, 0.5619999999999998, 0.5579999999999998, 0.5539999999999998, 
                0.5499999999999998, 0.5459999999999998, 0.5419999999999998, 0.5379999999999998, 0.5339999999999998, 0.5299999999999998, 0.5259999999999998, 
                0.5219999999999998, 0.5179999999999998, 0.5139999999999998, 0.5099999999999998, 0.5059999999999998, 0.5019999999999998, 0.4979999999999998, 
                0.4939999999999998, 0.48999999999999977, 0.48599999999999977, 0.48199999999999976, 0.47799999999999976, 0.47399999999999975, 0.46999999999999975, 
                0.46599999999999975, 0.46199999999999974, 0.45799999999999974, 0.45399999999999974, 0.44999999999999973, 0.44599999999999973, 0.4419999999999997, 
                0.4379999999999997, 0.4339999999999997, 0.4299999999999997, 0.4259999999999997, 0.4219999999999997, 0.4179999999999997, 0.4139999999999997, 
                0.4099999999999997, 0.4059999999999997, 0.4019999999999997, 0.3979999999999997, 0.3939999999999997, 0.3899999999999997, 0.3859999999999997, 
                0.3819999999999997, 0.37799999999999967, 0.37399999999999967, 0.36999999999999966, 0.36599999999999966, 0.36199999999999966, 0.35799999999999965, 
                0.35399999999999965, 0.34999999999999964, 0.34599999999999964, 0.34199999999999964, 0.33799999999999963, 0.33399999999999963, 0.3299999999999996, 
                0.3259999999999996, 0.3219999999999996, 0.3179999999999996, 0.3139999999999996, 
                0.3099999999999996, 0.3059999999999996, 0.3019999999999996, 0.2979999999999996, 0.2939999999999996, 0.2899999999999996, 0.2859999999999996, 
                0.2819999999999996, 0.2779999999999996, 0.2739999999999996, 0.2699999999999996
                ]
                

#model_path = './save/model.ckpt'

batchX_placeholder = tf.placeholder(tf.int32, [None, truncated_backprop_length])
batchX_placeholder_rev = tf.placeholder(tf.int32, [None, truncated_backprop_length])

batchY_placeholder = tf.placeholder(tf.int32, [None, num_classes]) # perhaps I am right ??
keep_probability = tf.placeholder(tf.float32)
keep_probability2 = tf.placeholder(tf.float32)
l_rate = tf.placeholder(tf.float32)

#batchX_placeholder_POS = tf.placeholder(tf.int32, [None, truncated_backprop_length])
#batchX_placeholder_POS_rev = tf.placeholder(tf.int32, [None, truncated_backprop_length])

#batchX_placeholder_WNET = tf.placeholder(tf.int32, [None, truncated_backprop_length])
#batchX_placeholder_WNET_rev = tf.placeholder(tf.int32, [None, truncated_backprop_length])

batchX_placeholder_GR = tf.placeholder(tf.int32, [None, truncated_backprop_length])
batchX_placeholder_GR_rev = tf.placeholder(tf.int32, [None, truncated_backprop_length])



Wordvec_embedings = tf.get_variable(name="Wordvec_embediings", shape=Embeddings.shape, initializer=tf.constant_initializer(Embeddings), trainable=True)
#POS_embeddings = tf.get_variable(name="POS_embeddings", shape=pos_embeddings.shape, initializer=tf.constant_initializer(pos_embeddings), trainable=True)
#Wnet_Embeddings = tf.get_variable(name="Wnet_embeddings", shape=wn_embeddings.shape, initializer=tf.constant_initializer(wn_embeddings), trainable=True)
GRel_Embeddings = tf.get_variable(name="GRel_Embeddings", shape=gr_embeddings.shape, initializer=tf.constant_initializer(gr_embeddings), trainable=True)



cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
cell1 = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)

#cell_POS_f = tf.contrib.rnn.BasicLSTMCell(pos_rec_1, state_is_tuple=True)
#cell_POS_r = tf.contrib.rnn.BasicLSTMCell(pos_rec_1, state_is_tuple=True)

#cell_WNET_f = tf.contrib.rnn.BasicLSTMCell(wn_rec_1, state_is_tuple=True)
#cell_WNET_r = tf.contrib.rnn.BasicLSTMCell(wn_rec_1, state_is_tuple=True)

cell_GR_f = tf.contrib.rnn.BasicLSTMCell(gr_rec_1, state_is_tuple=True)
cell_GR_r = tf.contrib.rnn.BasicLSTMCell(gr_rec_1, state_is_tuple=True)

input_forward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder)
input_backward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder_rev)

#input_POS_forward = tf.nn.embedding_lookup(POS_embeddings, batchX_placeholder_POS)
#input_POS_backward = tf.nn.embedding_lookup(POS_embeddings, batchX_placeholder_POS_rev)

#input_WNET_forward = tf.nn.embedding_lookup(Wnet_Embeddings, batchX_placeholder_WNET)
#input_WNET_backward = tf.nn.embedding_lookup(Wnet_Embeddings, batchX_placeholder_WNET_rev)

input_GR_forward = tf.nn.embedding_lookup(GRel_Embeddings, batchX_placeholder_GR)
input_GR_backward = tf.nn.embedding_lookup(GRel_Embeddings, batchX_placeholder_GR_rev)



Embedding_dropout_left = tf.nn.dropout(input_forward, keep_probability)
Embedding_dropout_right = tf.nn.dropout(input_backward, keep_probability) 

inputs_forward = [tf.squeeze(input_, [1]) for input_ in tf.split(Embedding_dropout_left, truncated_backprop_length, 1)]
inputs_backward = [tf.squeeze(input_, [1]) for input_ in tf.split(Embedding_dropout_right, truncated_backprop_length, 1)]

#inputs_forward_POS = [tf.squeeze(input_, [1]) for input_ in tf.split(input_POS_forward, truncated_backprop_length, 1)]
#inputs_backward_POS = [tf.squeeze(input_, [1]) for input_ in tf.split(input_POS_backward, truncated_backprop_length, 1)]

#inputs_forward_WNET = [tf.squeeze(input_, [1]) for input_ in tf.split(input_WNET_forward, truncated_backprop_length, 1)]
#inputs_backward_WNET = [tf.squeeze(input_, [1]) for input_ in tf.split(input_WNET_backward, truncated_backprop_length, 1)]

inputs_forward_GR = [tf.squeeze(input_, [1]) for input_ in tf.split(input_GR_forward, truncated_backprop_length, 1)]
inputs_backward_GR = [tf.squeeze(input_, [1]) for input_ in tf.split(input_GR_backward, truncated_backprop_length, 1)]

outputs_forward, state_forward = tf.nn.static_rnn(cell, inputs_forward, dtype=tf.float32, scope="LSTM1")
outputs_backward, state_backward = tf.nn.static_rnn(cell1, inputs_backward, dtype=tf.float32, scope="LSTM2")

#POS_outputs_f, POS_state_f = tf.contrib.rnn.static_rnn(cell_POS_f, inputs_forward_POS, dtype=tf.float32, scope="POS_LSTM1")
#POS_outputs_r, POS_state_r = tf.contrib.rnn.static_rnn(cell_POS_r, inputs_backward_POS, dtype=tf.float32, scope="POS_LSTM2")

#Wnet_outputs_f, Wnet_state_f = tf.contrib.rnn.static_rnn(cell_WNET_f, inputs_forward_WNET, dtype=tf.float32, scope="WNET_LSTM1")
#Wnet_outputs_r, Wnet_state_r = tf.contrib.rnn.static_rnn(cell_WNET_r, inputs_backward_WNET, dtype=tf.float32, scope="WNET_LSTM2")

GR_outputs_f, GR_state_f = tf.contrib.rnn.static_rnn(cell_GR_f, inputs_forward_GR, dtype=tf.float32, scope="GR_LSTM1")
GR_outputs_r, GR_state_r = tf.contrib.rnn.static_rnn(cell_GR_r, inputs_backward_GR, dtype=tf.float32, scope="GR_LSTM2")


'''
print(len(outputs_forward))
print(len(outputs_backward))
print(len(GR_outputs_f))
'''


for i in xrange(0, truncated_backprop_length):
    outputs_forward[i] = tf.expand_dims(outputs_forward[i], axis=1)
    outputs_backward[i] = tf.expand_dims(outputs_backward[i], axis=1)
    
    #POS_outputs_f[i] = tf.expand_dims(POS_outputs_f[i], axis=1)
    #POS_outputs_r[i] = tf.expand_dims(POS_outputs_r[i], axis=1)
    
    #Wnet_outputs_f[i] = tf.expand_dims(Wnet_outputs_f[i], axis=1)
    #Wnet_outputs_r[i] = tf.expand_dims(Wnet_outputs_r[i], axis=1)
    
    GR_outputs_f[i] = tf.expand_dims(GR_outputs_f[i], axis=1)
    GR_outputs_r[i] = tf.expand_dims(GR_outputs_r[i], axis=1)


forward_tensor_concat = tf.concat([outputs_forward[0], outputs_forward[1]], 1)
backward_tensor_concat = tf.concat([outputs_backward[0], outputs_backward[1]], 1)

#concat_POS_f = tf.concat([POS_outputs_f[0], POS_outputs_f[1]], 1)
#concat_POS_r = tf.concat([POS_outputs_r[0], POS_outputs_r[1]], 1)

#concat_Wnet_f = tf.concat([Wnet_outputs_f[0], Wnet_outputs_f[1]], 1)
#concat_Wnet_r = tf.concat([Wnet_outputs_r[0], Wnet_outputs_r[1]], 1)



concat_GR_f = tf.concat([GR_outputs_f[0], GR_outputs_f[1]], 1)
concat_GR_r = tf.concat([GR_outputs_r[0], GR_outputs_r[1]], 1)

for i in xrange(2, truncated_backprop_length):
    forward_tensor_concat = tf.concat([forward_tensor_concat, outputs_forward[i]], 1)
    backward_tensor_concat = tf.concat([backward_tensor_concat, outputs_backward[i]], 1)

    #concat_POS_f = tf.concat([concat_POS_f, POS_outputs_f[i]], 1)
    #concat_POS_r = tf.concat([concat_POS_r, POS_outputs_r[i]], 1)

    concat_GR_f = tf.concat([concat_GR_f, GR_outputs_f[i]], 1)
    concat_GR_r = tf.concat([concat_GR_r, GR_outputs_r[i]], 1)

    #concat_Wnet_f = tf.concat([concat_Wnet_f, Wnet_outputs_f[i]], 1)
    #concat_Wnet_r = tf.concat([concat_Wnet_r, Wnet_outputs_r[i]], 1)


#### Concat the ops and apply one cnn only, else do a dependency unit kind of thing or else apply 2 CNNs seperately on 2 channels 





forward_concat = tf.concat([forward_tensor_concat, concat_GR_f], 2)
backward_concat =  tf.concat([backward_tensor_concat, concat_GR_r], 2)

forward_concat1 = tf.reduce_max(forward_concat, 1)
backward_concat1 = tf.reduce_max(backward_concat, 1)

print(forward_concat)
print(backward_concat)



max_pooled = tf.concat([forward_concat1, backward_concat1], 1)

print(max_pooled)

#exit()



nb_filter1 = 200
nb_filter = 100 

W2 = tf.get_variable("Filter_SDP", shape=[5, 175, nb_filter1],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b2 = tf.get_variable(name="Bias_SDP", shape=[nb_filter1], initializer=tf.zeros_initializer())

W21 = tf.get_variable("Filter_SDP21", shape=[3, 175, nb_filter],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b21 = tf.get_variable(name="Bias_SDP21", shape=[nb_filter], initializer=tf.zeros_initializer())

output_cnv21 = tf.nn.conv1d(value=forward_concat, filters=W21, stride=1, padding='SAME')

output_cnv1 = tf.nn.conv1d(value=forward_concat, filters=W2, stride=1, padding='SAME')

h21 = tf.tanh(tf.nn.bias_add(output_cnv21, b21), name="Hyperbolic21")

h21 = tf.expand_dims(h21, axis=1)


h1 = tf.tanh(tf.nn.bias_add(output_cnv1, b2), name="Hyperbolic1")

h1 = tf.expand_dims(h1, axis=1)

        # Max-pooling over the outputs

pooled21 = tf.nn.max_pool(
            h21,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool21")

pooled21 = tf.squeeze(tf.squeeze(pooled21, 1), 1)

pooled1 = tf.nn.max_pool(
            h1,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool1")

pooled1 = tf.squeeze(tf.squeeze(pooled1, 1), 1)


#W3 = tf.get_variable("Filter_SDP2", shape=[5, 150, nb_filter1],
#           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
#b3 = tf.get_variable(name="Bias_SDP2", shape=[nb_filter2], initializer=tf.zeros_initializer())

output_cnv2 = tf.nn.conv1d(value=backward_concat, filters=W2, stride=1, padding='SAME')
output_cnv22 = tf.nn.conv1d(value=backward_concat, filters=W21, stride=1, padding='SAME')

h2 = tf.tanh(tf.nn.bias_add(output_cnv2, b2), name="Hyperbolic2")
h22 = tf.tanh(tf.nn.bias_add(output_cnv22, b21), name="Hyperbolic22")


h2 = tf.expand_dims(h2, axis=1)
h22 = tf.expand_dims(h22, axis=1)
        # Max-pooling over the outputs

pooled2 = tf.nn.max_pool(
            h2,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool2")

pooled22 = tf.nn.max_pool(
            h22,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool22")

pooled2 = tf.squeeze(tf.squeeze(pooled2, 1), 1)
pooled22 = tf.squeeze(tf.squeeze(pooled22, 1), 1)

W4 = tf.get_variable("W_soft", shape=[450, num_classes],
           initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=0.001), trainable=True)

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b4 = tf.get_variable(name="Bias_Softmax", shape=[num_classes], initializer=tf.zeros_initializer())


output_conc = tf.concat([pooled1, pooled21, pooled2, pooled22], 1)

output_conc = tf.nn.dropout(output_conc, keep_probability2)

output_dense = tf.layers.dense(output_conc, 100, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(), kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.001),
                            trainable=True
                        )

l = tf.trainable_variables()

l2_cost = 0.0 

for t in l:
    print (t.name)
    l2_cost += tf.nn.l2_loss(t) 

output_dense = tf.concat([output_dense, max_pooled] ,1)

logits = tf.matmul(output_dense, W4) + b4
predictions = tf.nn.softmax(logits)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batchY_placeholder))
cost = cost + 0.0000001 * l2_cost 
#train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
train_step = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(batchY_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

train_b = 0
train_e = 13913
valid_b = 0
valid_e = 800
test_b  = 0
test_e  = 2717


def build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e):

    SPTs_train = load_rev_valid_aug18class_samples_train(path_raw_data, file_train)
    SPTs_test  = load_rev_direc_samples_test(path_raw_data, file_test)
    
    # Do the padding as well 

    batchX_train_forward = []
    batchX_train_reverse = []
    for i in xrange(train_b, train_e):
        SPT = SPTs_train[i]
 
        # convert SPT[4]  and SPT[5] into word indices 

        train_ex = []
        for item in SPT[4]:
            word_index = vocab_dic[item][0]
            train_ex.append(word_index)

        train_ex_rev = []    
        for item in SPT[5]:
            word_index = vocab_dic[item][0]
            train_ex_rev.append(word_index)

        if len(train_ex) < truncated_backprop_length:
            start = len(train_ex)
            for i in range(start, truncated_backprop_length):
                train_ex.append(26463)

        if len(train_ex_rev) < truncated_backprop_length:
            start = len(train_ex_rev)
            for i in range(start, truncated_backprop_length):
                train_ex_rev.append(26463)

        batchX_train_forward.append(train_ex)
        batchX_train_reverse.append(train_ex_rev)
    
    
    batchX_test_forward = []
    batchX_test_reverse = []
    for k in xrange(test_b, test_e):
        SPT = SPTs_test[k]
 
        test_ex = []
        for item in SPT[4]:
            word_index = vocab_dic[item][0]
            test_ex.append(word_index)

        test_ex_rev = []    
        for item in SPT[5]:
            word_index = vocab_dic[item][0]
            test_ex_rev.append(word_index)

        if len(test_ex) < truncated_backprop_length:
            start = len(test_ex)
            for i in range(start, truncated_backprop_length):
                test_ex.append(26463)

        if len(test_ex_rev) < truncated_backprop_length:
            start = len(test_ex_rev)
            for i in range(start, truncated_backprop_length):
                test_ex_rev.append(26463)

        batchX_test_forward.append(test_ex)
        batchX_test_reverse.append(test_ex_rev)

    labels_train = load_rev_valid_aug18class_labels_train(path_raw_data, file_train)
    labels_test  = load_rev_direc_labels_test(path_raw_data, file_test)

    # make labels one hot 
    y_train = np.zeros(shape=(len(labels_train), 19))
    for i in xrange(0, len(labels_train)):
        one_hot = np.zeros(19)
        label = labels_train[i]
        one_hot[label] = 1
        y_train[i] = one_hot

    #print(y_train.shape)
    #print('test labels',labels_test)

    y_test = np.zeros(shape=(len(labels_test), 19))
    for i in xrange(0, len(labels_test)):
        one_hot = np.zeros(19)
        label = labels_test[i]
        one_hot[label] = 1
        y_test[i] = one_hot

    #print('test label unique', np.unique(np.array(labels_test)))

    return batchX_train_forward, batchX_train_reverse, y_train, batchX_test_forward, batchX_test_reverse, y_test 


def load_POS_GR_Wnet_train(train_b, train_e, valid_b, valid_e, test_b, test_e):

    SPTs_train = load_rev_valid_aug18class_samples_train(path_raw_data, file_train)
    SPTs_test  = load_rev_direc_samples_test(path_raw_data, file_test)
    
    # Do the padding as well 

    POS_batchX_train_f = []
    POS_batchX_train_r = []
    
    Wnet_batchX_train_f = []
    Wnet_batchX_train_r = []
    
    GR_batchX_train_f = []
    GR_batchX_train_r = []
    
    SPTs_train[2668] = [['palm-19'], ['palm-19', 'of-20', 'hand-22'], ['palm'], ['hand', 'of', 'palm'], ['palm'], ['hand', 'of', 'palm'], ['NN'], ['NN', 'IN', 'NN'], [], ['pobj', 'prep'], ['B-noun.body'], ['B-noun.body', '0', 'B-noun.body']]
    
    # print("pos_dic:", pos_dic)
    # print("wn_dic:", wn_dic)
    # print("gr_dic:", gr_dic)
    # exit(1)
    for i in xrange(train_b, train_e):
        SPT = SPTs_train[i]
        # convert SPT[4]  and SPT[5] into word indices 
        POS_train_ex_f = []
        
        for item in SPT[6]:
            if item == '$':
                item = 'paddd'
            POS_index = pos_dic[item]
            POS_train_ex_f.append(POS_index)

        POS_train_ex_r = []    
        for item in SPT[7]:
            if item == '$':
                item = 'paddd'
            
            POS_index = pos_dic[item]
            POS_train_ex_r.append(POS_index)

        if len(POS_train_ex_f) < truncated_backprop_length:
            start = len(POS_train_ex_f)
            for i in range(start, truncated_backprop_length):
                POS_train_ex_f.append(0)

        if len(POS_train_ex_r) < truncated_backprop_length:
            start = len(POS_train_ex_r)
            for i in range(start, truncated_backprop_length):
                POS_train_ex_r.append(0)


        POS_batchX_train_f.append(POS_train_ex_f)
        POS_batchX_train_r.append(POS_train_ex_r)
    
        Wnet_train_ex_f = []
        for item in SPT[10]:
            if item == 'I-noun.animals':
                item = 'I-noun.animal'
            elif item == 'B-noun-artifact' or item == ' B-noun.artifact' or ' B-noun.artifact ':
                item = 'B-noun.artifact' 
            elif item == 'B-noun.loaction':
                item = 'B-noun.location'      
            elif item == ' B-noun.time ':
                item = 'B-noun.time'
            elif item == ' B-noun.person':
                item = 'B-noun.person'
            Wnet_index = wn_dic[item]
            Wnet_train_ex_f.append(Wnet_index)

        Wnet_train_ex_r = []    
        for item in SPT[11]:
            if item == 'I-noun.animals':
                item = 'I-noun.animal'
            elif item == 'B-noun-artifact' or item == ' B-noun.artifact' or ' B-noun.artifact ':
                item = 'B-noun.artifact'    
            elif item == 'B-noun.loaction':
                item = 'B-noun.location'      
            elif item == ' B-noun.time ':
                item = 'B-noun.time'
            
            Wnet_index = wn_dic[item]
            Wnet_train_ex_r.append(Wnet_index)

        if len(Wnet_train_ex_f) < truncated_backprop_length:
            start = len(Wnet_train_ex_f)
            for i in range(start, truncated_backprop_length):
                Wnet_train_ex_f.append(0)

        if len(Wnet_train_ex_r) < truncated_backprop_length:
            start = len(Wnet_train_ex_r)
            for i in range(start, truncated_backprop_length):
                Wnet_train_ex_r.append(0)
    
        Wnet_batchX_train_f.append(Wnet_train_ex_f)
        Wnet_batchX_train_r.append(Wnet_train_ex_r)


        GR_train_ex_f = []
        for item in SPT[8]:
            GR_index = gr_dic[item]
            GR_train_ex_f.append(GR_index)

        GR_train_ex_r = []    
        for item in SPT[9]:
            GR_index = gr_dic[item]
            GR_train_ex_r.append(GR_index)

        if len(GR_train_ex_f) < truncated_backprop_length:
            start = len(GR_train_ex_f)
            for i in range(start, truncated_backprop_length):
                GR_train_ex_f.append(0)

        if len(GR_train_ex_r) < truncated_backprop_length:
            start = len(GR_train_ex_r)
            for i in range(start, truncated_backprop_length):
                GR_train_ex_r.append(0)
    
        GR_batchX_train_f.append(GR_train_ex_f)
        GR_batchX_train_r.append(GR_train_ex_r)

    POS_batchX_train_f = np.array(POS_batchX_train_f)
    POS_batchX_train_r = np.array(POS_batchX_train_r)
    Wnet_batchX_train_f = np.array(Wnet_batchX_train_f)
    Wnet_batchX_train_r = np.array(Wnet_batchX_train_r)    
    GR_batchX_train_f = np.array(GR_batchX_train_f)
    GR_batchX_train_r = np.array(GR_batchX_train_r)

    return POS_batchX_train_f, POS_batchX_train_r, Wnet_batchX_train_f, Wnet_batchX_train_r, GR_batchX_train_f, GR_batchX_train_r    


def load_POS_GR_Wnet_test(train_b, train_e, valid_b, valid_e, test_b, test_e):
    
    SPTs_train = load_rev_valid_aug18class_samples_train(path_raw_data, file_train)
    SPTs_test  = load_rev_direc_samples_test(path_raw_data, file_test)

    POS_batchX_test_f = []
    POS_batchX_test_r = []

    Wnet_batchX_test_f = []
    Wnet_batchX_test_r = []

    GR_batchX_test_f = []
    GR_batchX_test_r = []

    for k in xrange(test_b, test_e):
        SPT = SPTs_test[k]
 
        POS_test_ex_f = []
        for item in SPT[6]:
            if item == '$':
                item = 'paddd'
            pos_index = pos_dic[item]
            POS_test_ex_f.append(pos_index)

        POS_test_ex_r = []    
        for item in SPT[7]:
            if item == '$':
                item = 'paddd'
            pos_index = pos_dic[item]
            POS_test_ex_r.append(pos_index)

        if len(POS_test_ex_f) < truncated_backprop_length:
            start = len(POS_test_ex_f)
            for i in range(start, truncated_backprop_length):
                POS_test_ex_f.append(0)

        if len(POS_test_ex_r) < truncated_backprop_length:
            start = len(POS_test_ex_r)
            for i in range(start, truncated_backprop_length):
                POS_test_ex_r.append(0)

        POS_batchX_test_f.append(POS_test_ex_f)
        POS_batchX_test_r.append(POS_test_ex_r)

        Wnet_test_ex_f = []
        for item in SPT[10]:
            if item == 'I-noun.animals':
                item = 'I-noun.animal'
            elif item == 'B-noun-artifact' or item == ' B-noun.artifact' or ' B-noun.artifact ':
                item = 'B-noun.artifact'
            elif item == 'B-noun.loaction':
                item = 'B-noun.location'      
            elif item == ' B-noun.time ':
                item = 'B-noun.time'
            
            Wnet_index = wn_dic[item]
            Wnet_test_ex_f.append(Wnet_index)

        Wnet_test_ex_r = []    
        for item in SPT[11]:
            if item == 'I-noun.animals':
                item = 'I-noun.animal'
            elif item == 'B-noun-artifact' or item == ' B-noun.artifact' or ' B-noun.artifact ':
                item = 'B-noun.artifact'
            elif item == 'B-noun.loaction':
                item = 'B-noun.location'      
            elif item == ' B-noun.time ':
                item = 'B-noun.time'
            wnet_index = wn_dic[item]
            Wnet_test_ex_r.append(wnet_index)

        if len(Wnet_test_ex_f) < truncated_backprop_length:
            start = len(Wnet_test_ex_f)
            for i in range(start, truncated_backprop_length):
                Wnet_test_ex_f.append(0)

        if len(Wnet_test_ex_r) < truncated_backprop_length:
            start = len(Wnet_test_ex_r)
            for i in range(start, truncated_backprop_length):
                Wnet_test_ex_r.append(0)

        Wnet_batchX_test_f.append(Wnet_test_ex_f)
        Wnet_batchX_test_r.append(Wnet_test_ex_r)


        GR_test_ex_f = []
        for item in SPT[8]:
            GR_index = gr_dic[item]
            GR_test_ex_f.append(GR_index)

        GR_test_ex_r = []    
        for item in SPT[9]:
            GR_index = gr_dic[item]
            GR_test_ex_r.append(GR_index)

        if len(GR_test_ex_f) < truncated_backprop_length:
            start = len(GR_test_ex_f)
            for i in range(start, truncated_backprop_length):
                GR_test_ex_f.append(0)

        if len(GR_test_ex_r) < truncated_backprop_length:
            start = len(GR_test_ex_r)
            for i in range(start, truncated_backprop_length):
                GR_test_ex_r.append(0)

        GR_batchX_test_f.append(GR_test_ex_f)
        GR_batchX_test_r.append(GR_test_ex_r)

    POS_batchX_test_f = np.array(POS_batchX_test_f)
    POS_batchX_test_r = np.array(POS_batchX_test_r)
    Wnet_batchX_test_f = np.array(Wnet_batchX_test_f)
    Wnet_batchX_test_r = np.array(Wnet_batchX_test_r)    
    GR_batchX_test_f = np.array(GR_batchX_test_f)
    GR_batchX_test_r = np.array(GR_batchX_test_r)

    return POS_batchX_test_f, POS_batchX_test_r, Wnet_batchX_test_f, Wnet_batchX_test_r, GR_batchX_test_f, GR_batchX_test_r    

#################################################################################################################################
#
x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
POS_X_train_f, POS_X_train_r, Wnet_X_train_f, Wnet_X_train_r, GR_X_train_f,GR_X_train_r = load_POS_GR_Wnet_train(train_b, train_e, valid_b, valid_e, test_b, test_e)
POS_X_test_f, POS_X_test_r, Wnet_X_test_f, Wnet_X_test_r, GR_X_test_f, GR_X_test_r  =  load_POS_GR_Wnet_test(train_b, train_e, valid_b, valid_e, test_b, test_e)   

x_train_left = np.array(x_forward)
x_train_right = np.array(x_backward)
x_test_left = np.array(x_test_f)
x_test_right = np.array(x_test_r)

x_valid_left = x_train_left[13113:]
x_train_left = x_train_left[:13113]

x_valid_right = x_train_right[13113:]
x_train_right = x_train_right[:13113]

y_valid = y_train[13113:]
y_train = y_train[:13113]

GR_X_valid_f = GR_X_train_f[13113:]
GR_X_train_f = GR_X_train_f[:13113]
GR_X_valid_r = GR_X_train_r[13113:]
GR_X_train_r = GR_X_train_r[:13113]


POS_X_valid_f = POS_X_train_f[13113:]
POS_X_train_f = POS_X_train_f[:13113]
POS_X_valid_r = POS_X_train_r[13113:]
POS_X_train_r = POS_X_train_r[:13113]

saver = tf.train.Saver(max_to_keep=10000)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    

    accuracies = []
    valid_accuracies = []
    for epoch_idx in range(len(SGD_lrate3)):
        
        # shuffle the training data 
        shuffled_indices = np.random.permutation(np.arange(len(x_train_left)))
        shuffled_x_left = x_train_left[shuffled_indices]
        shuffled_x_right = x_train_right[shuffled_indices]
        
        #POS_shuffled_x_f = POS_X_train_f[shuffled_indices]
        #POS_shuffled_x_r = POS_X_train_r[shuffled_indices]
        #Wnet_shuffled_x_f = Wnet_X_train_f[shuffled_indices]
        #Wnet_shuffled_x_r = Wnet_X_train_r[shuffled_indices]
        
        GR_shuffled_x_f = GR_X_train_f[shuffled_indices]
        GR_shuffled_x_r = GR_X_train_r[shuffled_indices]
        
        shuffled_y = y_train[shuffled_indices]
        
        epoch_loss = 0
        num_batches = 262
        
        lr = SGD_lrate3[epoch_idx]
        print("New epoch", epoch_idx)
        print("Learning rate", lr)
        acc = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batchX_left = shuffled_x_left[start_idx:end_idx]
            batchX_right = shuffled_x_right[start_idx:end_idx]
            batchY = shuffled_y[start_idx:end_idx]

            #batchX_POS_f = POS_shuffled_x_f[start_idx:end_idx]
            #batchX_POS_r = POS_shuffled_x_r[start_idx:end_idx]
            #batchX_Wnet_f = Wnet_shuffled_x_f[start_idx:end_idx]
            #batchX_Wnet_r = Wnet_shuffled_x_r[start_idx:end_idx]
            
            batchX_GR_f = GR_shuffled_x_f[start_idx:end_idx]
            batchX_GR_r = GR_shuffled_x_r[start_idx:end_idx]
                           
            _predictions, _accuracy, _total_loss, _train_step = sess.run(
                [ predictions, accuracy, cost, train_step],
                feed_dict={
                    batchX_placeholder: batchX_left,
                    batchX_placeholder_rev: batchX_right,
                    batchY_placeholder: batchY,
                    batchX_placeholder_GR: batchX_GR_f,
                    batchX_placeholder_GR_rev: batchX_GR_r,
                    keep_probability: 0.5,
                    keep_probability2: 0.7,
                    l_rate:lr
                    
                })
            
            epoch_loss += _total_loss
            acc += _accuracy
            
          
        print("epoch loss", epoch_loss)        
        print('Epoch Accuracy', acc/num_batches)

        valid_predictions, valid_accuracy, valid_loss  = sess.run(
                [predictions, accuracy, cost],
                feed_dict={
                batchX_placeholder: x_valid_left,
                batchX_placeholder_rev: x_valid_right,
                batchY_placeholder: y_valid,
                keep_probability: 1.0,
                keep_probability2: 1.0,
                batchX_placeholder_GR: GR_X_valid_f,
                batchX_placeholder_GR_rev: GR_X_valid_r,
                l_rate: 0.001                    
                }
            )

        print('Validation Loss: ', valid_loss)        
        print('Validation accuracy :', valid_accuracy)
        
        valid_accuracies.append(valid_accuracy)
        
        val_preds = np.argmax(valid_predictions, 1)
        val_target = np.argmax(y_valid, 1)
        val_target = np.add(val_target, 1)

        #with open('rec_cnv_3_2_valid_target', 'w') as f:
        #    for item in val_target:
        #        f.write("%s\n" % item)
        #print(val_preds.shape)
        
        val_preds = np.add(val_preds, 1)
        
        with open('rec_cnv3_filter_valid_predictions', 'w') as f:
            for item in val_preds:
                f.write("%s\n" % item)
        
        os.system('python torch_eval_valid.py rec_cnv3_filter_valid_predictions valid_targets %d'%epoch_idx)


        
        test_predictions, test_accuracy, test_loss  = sess.run(
                [predictions, accuracy, cost],
                feed_dict={
                batchX_placeholder: x_test_left,
                batchX_placeholder_rev: x_test_right,
                batchY_placeholder: y_test,
                keep_probability: 1.0,
                keep_probability2: 1.0,
                batchX_placeholder_GR: GR_X_test_f,
                batchX_placeholder_GR_rev: GR_X_test_r,
                l_rate: 0.001                    
                }
            )

        model_path = './save/model.ckpt' + str(epoch_idx)
        print(model_path)
        saver.save(sess,model_path)
        


        print('Test Loss: ', test_loss)        
        print('test accuracy :', test_accuracy)
        accuracies.append(test_accuracy)
        target = np.argmax(y_test, 1)
        preds = np.argmax(test_predictions, 1)
        print(preds.shape)
        preds = np.add(preds, 1)
        with open('rec_cnv3_filter_test_predictions', 'w') as f:
            for item in preds:
                f.write("%s\n" % item)
        #with open('tf_rnn_cpoy6_copy_targets', 'w') as f:
        #    for item in target:
        #        f.write("%s\n" % item)        
        os.system('python torch_eval.py rec_cnv3_filter_test_predictions test_target_modified %d'%epoch_idx)
        print('--------------------------------------------')
        print('\n')
    print(accuracies)    
            
