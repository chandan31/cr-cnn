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
#from tensorflow.contrib.layers import initializers

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
pos_rec_1 = 40
pos_rec_2 = 50
pos_rec_3 = 50
pos_rec_4 = 50

wn_self  = 50
wn_rec_1 = 40
wn_rec_2 = 50
wn_rec_3 = 50
wn_rec_4 = 50

gr_self  = 50
gr_rec_1 = 40
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



    
#learning_rate = 0.001
learning_rate = [0.27,0.243,0.2187, 0.19683, 0.177147, 0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 
                0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 0.0500315, 0.0450284, 0.0405255, 0.036473, 
                0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 0.0174449, 0.0157004, 0.0141304, 
                0.0141304, 0.0114456, 0.0114456, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0038356, 0.0033009,0.0028340, 0.0023485, 0.0018736, 0.0013567, 
                0.0012222, 0.001, 0.001, 0.001]
learning_rate = np.array(learning_rate)

file_train = "sem_train_8000.txt"
file_test = "sem_test_2717.txt"

vocab_dic  = load_wrd_vec_dic_v2(path_raw_data)

vocab_dic['paddd'] = [len(vocab_dic), np.zeros(200)] 

emb_lst    = load_emd_lst_v2(path_raw_data)
emb_lst.append(np.zeros(200))
Embeddings = np.array(emb_lst)


wn_num  = 50
pos_num = 15
gr_num  = 19
wn_dic  = lst_2_dic(WordNet_44_categories)
pos_dic = lst_2_dic(POS_15_categories)
gr_dic  = lst_2_dic(GR_19_categories)

wn_embeddings = np.random.randn(45, 50)
pos_embeddings = np.random.randn(16, 50)
gr_embeddings = np.random.randn(20,50)

wn_dic['paddd'] = 0
pos_dic['paddd'] = 0
gr_dic['paddd'] = 0
wn_embeddings[0] = np.zeros(50)
pos_embeddings[0] = np.zeros(50)
gr_embeddings[0] = np.zeros(50) 

##############################
'''
Model: LSTM with first  layer of size 128  
'''

num_epochs = 100
truncated_backprop_length = 15
state_size = 150
num_classes = 19
batch_size = 50


#################################################################################################################
#  Placeholder for RNN-1 (word_vecs)
#################################################################################################################
batchX_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchX_placeholder_rev = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes]) # perhaps I am right ??
#keep_probability = tf.placeholder(tf.float32)
#l_rate = tf.placeholder(tf.float32)

# define cell state and placeholders to feed the cell state 
#cell_state_forward = tf.placeholder(tf.float32, [batch_size, state_size])
#hidden_state_forward = tf.placeholder(tf.float32, [batch_size, state_size])

#init_state_forward = tf.contrib.rnn.LSTMStateTuple(cell_state_forward, hidden_state_forward)


#cell_state_backward = tf.placeholder(tf.float32, [batch_size, state_size])
#hidden_state_backward = tf.placeholder(tf.float32, [batch_size, state_size])
#init_state_backward = tf.contrib.rnn.LSTMStateTuple(cell_state_backward, hidden_state_backward)

#################################################################################################################
# Placeholder for RNN-1 (POS tags)
#################################################################################################################

batchX_placeholder_POS = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchX_placeholder_POS_rev = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

# we will need only one Y_placeholder because there is only one target 

#batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes]) # perhaps I am right ??
#keep_probability = tf.placeholder(tf.float32) Not required because we are not dropping out POS embeddings
#l_rate = tf.placeholder(tf.float32)

# define cell state and placeholders to feed the cell state 
#cell_state_POS_forward = tf.placeholder(tf.float32, [batch_size, pos_rec_1])
#hidden_state_POS_forward = tf.placeholder(tf.float32, [batch_size, pos_rec_1])

#init_state_POS_forward = tf.contrib.rnn.LSTMStateTuple(cell_state_POS_forward, hidden_state_POS_forward)


#cell_state_POS_backward = tf.placeholder(tf.float32, [batch_size, pos_rec_1])
#hidden_state_POS_backward = tf.placeholder(tf.float32, [batch_size, pos_rec_1])
#init_state_POS_backward = tf.contrib.rnn.LSTMStateTuple(cell_state_POS_backward, hidden_state_POS_backward)

##################################################################################################################
# Placeholder for Wnet 
#################################################################################################################
batchX_placeholder_WNET = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchX_placeholder_WNET_rev = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])


# define cell state and placeholders to feed the cell state 
#cell_state_WNET_forward = tf.placeholder(tf.float32, [batch_size, wn_rec_1])
#hidden_state_WNET_forward = tf.placeholder(tf.float32, [batch_size, wn_rec_1])

#init_state_WNET_forward = tf.contrib.rnn.LSTMStateTuple(cell_state_WNET_forward, hidden_state_WNET_forward)


#cell_state_WNET_backward = tf.placeholder(tf.float32, [batch_size, wn_rec_1])
#hidden_state_WNET_backward = tf.placeholder(tf.float32, [batch_size, wn_rec_1])
#init_state_WNET_backward = tf.contrib.rnn.LSTMStateTuple(cell_state_WNET_backward, hidden_state_WNET_backward)

##################################################################################################################
# PLaceholders for GR 
##################################################################################################################
batchX_placeholder_GR = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchX_placeholder_GR_rev = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])


# define cell state and placeholders to feed the cell state 
#cell_state_GR_forward = tf.placeholder(tf.float32, [batch_size, gr_rec_1])
#hidden_state_GR_forward = tf.placeholder(tf.float32, [batch_size, gr_rec_1])

#init_state_GR_forward = tf.contrib.rnn.LSTMStateTuple(cell_state_GR_forward, hidden_state_GR_forward)


#cell_state_GR_backward = tf.placeholder(tf.float32, [batch_size, gr_rec_1])
#hidden_state_GR_backward = tf.placeholder(tf.float32, [batch_size, gr_rec_1])
#init_state_GR_backward = tf.contrib.rnn.LSTMStateTuple(cell_state_GR_backward, hidden_state_GR_backward)

#################################################################################################################
# Embedding for Wordvecs
#################################################################################################################
Wordvec_embedings = tf.get_variable(name="Wordvec_embedings", shape=Embeddings.shape, initializer=tf.constant_initializer(Embeddings), trainable=True)

POS_embeddings = tf.get_variable(name="POS_embeddings", shape=pos_embeddings.shape, initializer=tf.constant_initializer(pos_embeddings), trainable=True)

Wnet_Embeddings = tf.get_variable(name="Wnet_embeddings", shape=wn_embeddings.shape, initializer=tf.constant_initializer(wn_embeddings), trainable=True)

GRel_Embeddings = tf.get_variable(name="GRel_Embeddings", shape=gr_embeddings.shape, initializer=tf.constant_initializer(gr_embeddings), trainable=True)

######################################################################################################
# softmax weights since we are working on concatenated state hence 2*state_size dimension 
######################################################################################################
W2 = tf.Variable(np.random.rand(100, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

##########################################################################################
# Define a total of 8 cells 2 for each channel 
#########################################################################################
cell = tf.contrib.rnn.BasicLSTMCell(state_size )
cell1 = tf.contrib.rnn.BasicLSTMCell(state_size)

cell_POS_f = tf.contrib.rnn.BasicLSTMCell(pos_rec_1)
cell_POS_r = tf.contrib.rnn.BasicLSTMCell(pos_rec_1)

cell_WNET_f = tf.contrib.rnn.BasicLSTMCell(wn_rec_1)
cell_WNET_r = tf.contrib.rnn.BasicLSTMCell(wn_rec_1)

cell_GR_f = tf.contrib.rnn.BasicLSTMCell(gr_rec_1)
cell_GR_r = tf.contrib.rnn.BasicLSTMCell(gr_rec_1)

###########################################################################################

# build the model and return the logits 
#Build the core computation graph, from the inputs to the logits.
# Input -> Embeddings [batch, Sentence_length, embedding]

########################################################################################################
# Inputs after Embedding Lookup 
########################################################################################################
input_forward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder)
input_backward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder_rev)

input_POS_forward = tf.nn.embedding_lookup(POS_embeddings, batchX_placeholder_POS)
input_POS_backward = tf.nn.embedding_lookup(POS_embeddings, batchX_placeholder_POS_rev)

input_WNET_forward = tf.nn.embedding_lookup(Wnet_Embeddings, batchX_placeholder_WNET)
input_WNET_backward = tf.nn.embedding_lookup(Wnet_Embeddings, batchX_placeholder_WNET_rev)

input_GR_forward = tf.nn.embedding_lookup(GRel_Embeddings, batchX_placeholder_GR)
input_GR_backward = tf.nn.embedding_lookup(GRel_Embeddings, batchX_placeholder_GR_rev)

############################################################################################################
# Dropout Only word embeddings (or all 4 chaneels embeddings??)
############################################################################################################

Embedding_dropout_left = tf.nn.dropout(input_forward, 0.5)
Embedding_dropout_right = tf.nn.dropout(input_backward, 0.5)

################################################################################################################ 
# Transform Input Tensor into List required by tf.rnn (does it require List or tensor ? print it)
################################################################################################################

inputs_forward = [tf.squeeze(input_, [1]) for input_ in tf.split(Embedding_dropout_left, truncated_backprop_length, 1)]
inputs_backward = [tf.squeeze(input_, [1]) for input_ in tf.split(Embedding_dropout_right, truncated_backprop_length, 1)]

inputs_forward_POS = [tf.squeeze(input_, [1]) for input_ in tf.split(input_POS_forward, truncated_backprop_length, 1)]
inputs_backward_POS = [tf.squeeze(input_, [1]) for input_ in tf.split(input_POS_backward, truncated_backprop_length, 1)]

inputs_forward_WNET = [tf.squeeze(input_, [1]) for input_ in tf.split(input_WNET_forward, truncated_backprop_length, 1)]
inputs_backward_WNET = [tf.squeeze(input_, [1]) for input_ in tf.split(input_WNET_backward, truncated_backprop_length, 1)]

inputs_forward_GR = [tf.squeeze(input_, [1]) for input_ in tf.split(input_GR_forward, truncated_backprop_length, 1)]
inputs_backward_GR = [tf.squeeze(input_, [1]) for input_ in tf.split(input_GR_backward, truncated_backprop_length, 1)]
##################################################################################################################

# print(' after embedding ',inputs)

#################################################################################################################
# Feed to Stacked LSTM 
# Expand the LSTM Network
#################################################################################################################
outputs_forward, state_forward = tf.contrib.rnn.static_rnn(cell, inputs_forward , dtype=tf.float32, scope="LSTM1")
outputs_backward, state_backward = tf.contrib.rnn.static_rnn(cell1, inputs_backward ,dtype=tf.float32, scope="LSTM2")

POS_outputs_f, POS_state_f = tf.contrib.rnn.static_rnn(cell_POS_f, inputs_forward_POS , dtype=tf.float32, scope="POS_LSTM1")
POS_outputs_r, POS_state_r = tf.contrib.rnn.static_rnn(cell_POS_r, inputs_backward_POS , dtype=tf.float32, scope="POS_LSTM2")

Wnet_outputs_f, Wnet_state_f = tf.contrib.rnn.static_rnn(cell_WNET_f, inputs_forward_WNET , dtype=tf.float32, scope="WNET_LSTM1")
Wnet_outputs_r, Wnet_state_r = tf.contrib.rnn.static_rnn(cell_WNET_r, inputs_backward_WNET , dtype=tf.float32, scope="WNET_LSTM2")

GR_outputs_f, GR_state_f = tf.contrib.rnn.static_rnn(cell_GR_f, inputs_forward_GR , dtype=tf.float32, scope="GR_LSTM1")
GR_outputs_r, GR_state_r = tf.contrib.rnn.static_rnn(cell_GR_r, inputs_backward_GR,  dtype=tf.float32, scope="GR_LSTM2")

#######################################################################################################################
# Build and return final logits (w/o softmax)
#######################################################################################################################
forward_op = state_forward[-1]
backward_op = state_backward[-1]

################################################################################################ 
# Cocat the Cell and hidden state and then apply softmax (Will make the model more complex)
# and hence difficult to train
################################################################################################


####################################################### 
# MAXPOOLING Over time                                #
#######################################################

# Concatenate all the tensors to form a single tensor of form 50 * 15 * state_size

for i in xrange(0, truncated_backprop_length):
    outputs_forward[i] = tf.expand_dims(outputs_forward[i], axis=1)
    outputs_backward[i] = tf.expand_dims(outputs_backward[i], axis=1)
    
    POS_outputs_f[i] = tf.expand_dims(POS_outputs_f[i], axis=1)
    POS_outputs_r[i] = tf.expand_dims(POS_outputs_r[i], axis=1)
    
    Wnet_outputs_f[i] = tf.expand_dims(Wnet_outputs_f[i], axis=1)
    Wnet_outputs_r[i] = tf.expand_dims(Wnet_outputs_r[i], axis=1)
    
    GR_outputs_f[i] = tf.expand_dims(GR_outputs_f[i], axis=1)
    GR_outputs_r[i] = tf.expand_dims(GR_outputs_r[i], axis=1)



forward_tensor_concat = tf.concat([outputs_forward[0], outputs_forward[1]], 1)
backward_tensor_concat = tf.concat([outputs_backward[0], outputs_backward[1]], 1)

concat_POS_f = tf.concat([POS_outputs_f[0], POS_outputs_f[1]], 1)
concat_POS_r = tf.concat([POS_outputs_r[0], POS_outputs_r[1]], 1)

concat_Wnet_f = tf.concat([Wnet_outputs_f[0], Wnet_outputs_f[1]], 1)
concat_Wnet_r = tf.concat([Wnet_outputs_r[0], Wnet_outputs_r[1]], 1)

concat_GR_f = tf.concat([GR_outputs_f[0], GR_outputs_f[1]], 1)
concat_GR_r = tf.concat([GR_outputs_r[0], GR_outputs_r[1]], 1)


for i in xrange(2, truncated_backprop_length):
    forward_tensor_concat = tf.concat([forward_tensor_concat, outputs_forward[i]], 1)
    backward_tensor_concat = tf.concat([backward_tensor_concat, outputs_backward[i]], 1)

    concat_POS_f = tf.concat([concat_POS_f, POS_outputs_f[i]], 1)
    concat_POS_r = tf.concat([concat_POS_r, POS_outputs_r[i]], 1)

    concat_GR_f = tf.concat([concat_GR_f, GR_outputs_f[i]], 1)
    concat_GR_r = tf.concat([concat_GR_r, GR_outputs_r[i]], 1)

    concat_Wnet_f = tf.concat([concat_Wnet_f, Wnet_outputs_f[i]], 1)
    concat_Wnet_r = tf.concat([concat_Wnet_r, Wnet_outputs_r[i]], 1)

max_pooled_forward = tf.reduce_max(forward_tensor_concat, 1)
max_pooled_backward = tf.reduce_max(backward_tensor_concat, 1)

POS_max_pooled_f = tf.reduce_max(concat_POS_f, 1)
POS_max_pooled_r = tf.reduce_max(concat_POS_r, 1)

Wnet_max_pooled_f = tf.reduce_max(concat_Wnet_f, 1)
Wnet_max_pooled_r = tf.reduce_max(concat_Wnet_r, 1)

GR_max_pooled_f = tf.reduce_max(concat_GR_f, 1)
GR_max_pooled_r = tf.reduce_max(concat_GR_r, 1)

###########################################################################
# MAXPOOLING done. 
###########################################################################


output_conc = tf.concat([max_pooled_forward, max_pooled_backward, POS_max_pooled_f, POS_max_pooled_r, 
                        Wnet_max_pooled_f, Wnet_max_pooled_r, GR_max_pooled_f, GR_max_pooled_r], 1)


##############################################################################
#   Fully connected layer with dropout rate 0.3 
##############################################################################
after_dropout = tf.nn.dropout(output_conc, 0.7)

output_fully_maxpooled = tf.contrib.layers.fully_connected(after_dropout,
    100,
    activation_fn=tf.nn.tanh,
    weights_initializer=tf.contrib.layers.xavier_initializer(),
    biases_initializer=tf.zeros_initializer(),
    trainable=True
    )


 
logits = tf.matmul(output_fully_maxpooled, W2) + b2
predictions = tf.nn.softmax(logits)


##############################################################################################
#  L2 Regularization added. 
##############################################################################################
train_vars_list = tf.trainable_variables()
lambda_l2 = 1e-5

# Frobenius normal form
l2_penalty = 0.0 
for w in train_vars_list:
    l2_penalty += tf.nn.l2_loss(w)

l2_penalty = lambda_l2*l2_penalty

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=batchY_placeholder) + l2_penalty)
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

################################################################################################
#
################################################################################################

# define op to calculate F-1 score on test data 

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
    maxm = 0
    for i in xrange(train_b, train_e):
        SPT = SPTs_train[i]
 
        # convert SPT[4]  and SPT[5] into word indices 
        #print(SPT)
        #maxm = max(maxm, max(len(SPT[4]),len(SPT[5]))) 
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
    
    # print(maxm) its 11 
    
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


#################################################################################################################
#  Train the Model -------------------------
#################################################################################################################

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.trainable_variables())
    train_vars_list = sess.run(tf.trainable_variables())
    for item in train_vars_list:
        print(item.shape)
    
    loss_list = []
    
    #_current_cell_state_forward = np.zeros((batch_size, state_size))
    #_current_cell_state_backward = np.zeros((batch_size, state_size))
    #_current_hidden_state_forward = np.zeros((batch_size, state_size))
    #_current_hidden_state_backward = np.zeros((batch_size, state_size))

    #_current_cell_state_POS_forward = np.zeros((batch_size, pos_rec_1))
    #_current_cell_state_POS_backward = np.zeros((batch_size, pos_rec_1))
    #_current_hidden_state_POS_forward = np.zeros((batch_size, pos_rec_1))
    #_current_hidden_state_POS_backward = np.zeros((batch_size, pos_rec_1))


    #_current_cell_state_Wnet_forward = np.zeros((batch_size, wn_rec_1))
    #_current_cell_state_Wnet_backward = np.zeros((batch_size, wn_rec_1))
    #_current_hidden_state_Wnet_forward = np.zeros((batch_size, wn_rec_1))
    #_current_hidden_state_Wnet_backward = np.zeros((batch_size, wn_rec_1))

    #_current_cell_state_GR_forward = np.zeros((batch_size, gr_rec_1))
    #_current_cell_state_GR_backward = np.zeros((batch_size, gr_rec_1))
    #_current_hidden_state_GR_forward = np.zeros((batch_size, gr_rec_1))
    #_current_hidden_state_GR_backward = np.zeros((batch_size, gr_rec_1))

    
    x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
    
    POS_X_train_f, POS_X_train_r, Wnet_X_train_f, Wnet_X_train_r, GR_X_train_f,GR_X_train_r = load_POS_GR_Wnet_train(train_b, train_e, valid_b, valid_e, test_b, test_e)
    POS_X_test_f, POS_X_test_r, Wnet_X_test_f, Wnet_X_test_r, GR_X_test_f, GR_X_test_r  =  load_POS_GR_Wnet_test(train_b, train_e, valid_b, valid_e, test_b, test_e)   
    
    x_train_left = np.array(x_forward)
    x_train_right = np.array(x_backward)
    x_test_left = np.array(x_test_f)
    x_test_right = np.array(x_test_r)


    accuracies = []
    for epoch_idx in range(num_epochs):
        

        # shuffle the training data 
        shuffled_indices = np.random.permutation(np.arange(len(x_forward)))
        
        shuffled_x_left = x_train_left[shuffled_indices]
        shuffled_x_right = x_train_right[shuffled_indices]

        POS_shuffled_x_f = POS_X_train_f[shuffled_indices]
        POS_shuffled_x_r = POS_X_train_r[shuffled_indices]
        Wnet_shuffled_x_f = Wnet_X_train_f[shuffled_indices]
        Wnet_shuffled_x_r = Wnet_X_train_r[shuffled_indices]
        GR_shuffled_x_f = GR_X_train_f[shuffled_indices]
        GR_shuffled_x_r = GR_X_train_r[shuffled_indices]
        
        # only one Y placeholder needed 
        shuffled_y = y_train[shuffled_indices]
        epoch_loss = 0
        # Better alternative ??
        # Do I need namescope ?? nope 
        num_batches = 278
        #print('test label unique', np.unique(np.argmax(y_test, 1))) working fine 19 labels 

        print("New epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batchX_left = shuffled_x_left[start_idx:end_idx]
            batchX_right = shuffled_x_right[start_idx:end_idx]
            batchY = shuffled_y[start_idx:end_idx]
            batchX_POS_f = POS_shuffled_x_f[start_idx:end_idx]
            batchX_POS_r = POS_shuffled_x_r[start_idx:end_idx]
            batchX_Wnet_f = Wnet_shuffled_x_f[start_idx:end_idx]
            batchX_Wnet_r = Wnet_shuffled_x_r[start_idx:end_idx]
            batchX_GR_f = GR_shuffled_x_f[start_idx:end_idx]
            batchX_GR_r = GR_shuffled_x_r[start_idx:end_idx]
          

            _total_loss, _train_step, _predictions  = sess.run(
                            [cost, train_step, predictions],
                            feed_dict={
                                batchX_placeholder: batchX_left,
                                batchX_placeholder_rev: batchX_right,
                                batchX_placeholder_POS: batchX_POS_f,
                                batchX_placeholder_POS_rev: batchX_POS_r,
                                batchX_placeholder_WNET: batchX_Wnet_f,
                                batchX_placeholder_WNET_rev: batchX_Wnet_r,
                                batchX_placeholder_GR: batchX_GR_f,
                                batchX_placeholder_GR_rev: batchX_GR_r,
                                batchY_placeholder: batchY
                                
                            })
            #print('batch label unique', np.unique(np.argmax(batchY, 1))) 19 labels fine 

           
            

            
            epoch_loss += _total_loss

            loss_list.append(_total_loss)

            if batch_idx%40 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
        print("epoch loss", epoch_loss)        



    

    
        test_batches = 54
        test_accuracy = 0
        total_test_preds = np.zeros(shape=(1, 19))
        last_test = y_test[-1]
        y_test[0] = last_test
        for batch_idx in range(test_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            test_batchX_left = x_test_left[start_idx:end_idx]
            test_batchX_right = x_test_right[start_idx:end_idx]
            batchX_POS_test_f = POS_X_test_f[start_idx:end_idx]
            batchX_POS_test_r = POS_X_test_r[start_idx:end_idx]
            batchX_Wnet_test_f = Wnet_X_test_f[start_idx:end_idx]
            batchX_Wnet_test_r = Wnet_X_test_r[start_idx:end_idx]
            batchX_GR_test_f = GR_X_test_f[start_idx:end_idx]
            batchX_GR_test_r = GR_X_test_r[start_idx:end_idx]
            
            
            test_batchY = y_test[start_idx:end_idx]
            #print(test)
            test_predictions, _accuracy,  = sess.run(
                    [predictions, accuracy],
                    feed_dict={
                        batchX_placeholder: test_batchX_left,
                        batchX_placeholder_rev: test_batchX_right,
                        batchX_placeholder_POS: batchX_POS_test_f,
                        batchX_placeholder_POS_rev: batchX_POS_test_r,
                        batchX_placeholder_WNET: batchX_Wnet_test_f,
                        batchX_placeholder_WNET_rev: batchX_Wnet_test_r,
                        batchX_placeholder_GR: batchX_GR_test_f,
                        batchX_placeholder_GR_rev: batchX_GR_test_r,
                        batchY_placeholder: test_batchY
                        

                    })

            
            



            #total_test_preds.append(test_predictions)
            test_predictions = np.array(test_predictions)
            #print('test label unique', np.unique(np.argmax(test_batchY, 1)))
            #print(test_predictions.shape)
            total_test_preds = np.concatenate((total_test_preds, test_predictions) , axis=0)
            #print(total_test_preds.shape)
            
            test_accuracy +=  _accuracy
            #print('Accuracy batch', _accuracy)

        
        print('test accuracy :', test_accuracy/test_batches)
        accuracies.append(test_accuracy/test_batches)
        #print(total_test_preds)
        # So I have the Predictions on all test data. Now I need to calcultae classwise TP and TN FN etc 
        # true positives 

        total_test_preds = np.array(total_test_preds)
        y_test_m = y_test[:2700]
        total_test_preds_m = total_test_preds[1:]
        #print('test label unique in modified test set', np.unique(np.argmax(y_test_m, 1)))
        #print('Total Predistions',total_test_preds_m.shape)
        #print('Y test shape', y_test_m.shape)

        c1 = np.argmax(y_test_m, 1)
        c2 = np.argmax(total_test_preds_m, 1)
        #c1_new = c1.reshape(2700,1)
        #c2_new = c2.reshape(2700,1)
        
        #print('c1', c1_new.shape)
        #print('c2', c2_new.shape)
        #print('Unique C1:',np.unique(c1_new).shape)
        #print('Unique C2:',np.unique(c2_new).shape)

        confus_matrix = confusion_matrix(c1, c2) # labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
        
        True_positive = np.zeros(len(confus_matrix[0]))
        False_negative = np.zeros(len(confus_matrix[0]))
        False_positive = np.zeros(len(confus_matrix[0]))

        print('False Positive shape', True_positive.shape)
        
        
        row_sum = np.sum(confus_matrix, axis=1)
        column_sum = np.sum(confus_matrix, axis=0)

        #print('Column Sum', column_sum)

        for i in xrange(0, 19):
            True_positive[i] = confus_matrix[i][i]
            if True_positive[i] == 0.0:
                True_positive[i] = 1e-6
            False_negative[i] = row_sum[i] - True_positive[i]
            if False_negative[i] == 0.0:
                False_negative[i] = 1e-6
            False_positive[i] = column_sum[i] - True_positive[i]
            if False_positive[i] == 0.0:
                False_positive[i] = 1e-6


        print(confus_matrix)    
        
        for i in xrange(0, 19):
            print(column_sum[i] - True_positive[i])

        for i in xrange(0, 19):
            print('TP:', True_positive[i])
            print('FN:', False_negative[i])
            print('FP:', False_positive[i])
        
        
            
        # calculate F-1 score  
        # Precision first 
        Precision_m = 0.0
        for i in xrange(0, len(True_positive) - 1):
            Precision_m += (True_positive[i]) / (True_positive[i] + False_positive[i])

        Precision_m = (Precision_m)/(num_classes -1)
        
        # Recall 
        Recall_m = 0.0
        for i in xrange(0, len(True_positive) - 1):
            Recall_m += (True_positive[i]) / (True_positive[i] + False_negative[i])

        Recall_m = (Recall_m) / (num_classes - 1)  # ignore the other class     

        F1_score = (2 * (Precision_m * Recall_m)) / (Precision_m + Recall_m)

        print(Precision_m)
        print(Recall_m)
        print(F1_score)
        
    print(accuracies)    
        #_current_cell_state_forward, _current_hidden_state_forward = _current_state_forward
        #_current_cell_state_backward, _current_hidden_state_backward = _current_state_backward
            

               


