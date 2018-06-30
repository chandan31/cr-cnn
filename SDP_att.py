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

##############################
'''
Model: LSTM with first  layer of size 128  
'''

num_epochs = 120
truncated_backprop_length = 12
state_size = 150
num_classes = 19
batch_size = 50

SGD_lrate1 = [0.55, 0.54, 0.536, 0.53, 0.525, 0.521, 0.53, 0.516, 0.51, 0.50, 0.48, 0.477, 0.460, 0.445, 0.433, 0.428, 0.420, 0.39, 0.36, 0.348, 0.32, 0.31, 
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


batchX_placeholder = tf.placeholder(tf.int32, [None, truncated_backprop_length])
batchX_placeholder_rev = tf.placeholder(tf.int32, [None, truncated_backprop_length])

batchY_placeholder = tf.placeholder(tf.int32, [None, num_classes]) # perhaps I am right ??
keep_probability = tf.placeholder(tf.float32)
l_rate = tf.placeholder(tf.float32)

Wordvec_embedings = tf.get_variable(name="Wordvec_embediings", shape=Embeddings.shape, initializer=tf.constant_initializer(Embeddings), trainable=True)


W2 = tf.Variable(np.random.rand(2*state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
cell1 = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)

input_forward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder)
input_backward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder_rev)

Embedding_dropout_left = tf.nn.dropout(input_forward, keep_probability)
Embedding_dropout_right = tf.nn.dropout(input_backward, keep_probability) 

inputs_forward = [tf.squeeze(input_, [1]) for input_ in tf.split(Embedding_dropout_left, truncated_backprop_length, 1)]
inputs_backward = [tf.squeeze(input_, [1]) for input_ in tf.split(Embedding_dropout_right, truncated_backprop_length, 1)]


outputs_forward, state_forward = tf.nn.static_rnn(cell, inputs_forward, dtype=tf.float32, scope="LSTM1")
outputs_backward, state_backward = tf.nn.static_rnn(cell1, inputs_backward, dtype=tf.float32, scope="LSTM2")


####################################################### 
# Soft Attenttion Over time                                #
#######################################################


H = tf.stack([h for h in outputs_forward])

H = tf.transpose(H, [1, 0, 2])

M = tf.tanh(H)
att_w = tf.get_variable('A_W', shape=[state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=True)
alpha = tf.nn.softmax(tf.tensordot(M, att_w, axes=(2,0)))

alpha1 = tf.stack([alpha for i in xrange(state_size)])
r = tf.reduce_sum(tf.multiply(M, tf.transpose(alpha1,[1, 2, 0])), 1)
forward_rep = tf.tanh(r)

print(forward_rep)

# Do I need 2 att_w vectors ?

H1 = tf.stack([h1 for h1 in outputs_backward])
H1 = tf.transpose(H1, [1, 0, 2])


M1 = tf.tanh(H1)
att_w1 = tf.get_variable('A_W1', shape=[state_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32, trainable=True)
alpha2 = tf.nn.softmax(tf.tensordot(M1, att_w1, axes=(2,0)))
alpha21 = tf.stack([alpha2 for i in xrange(state_size)])
r1 = tf.reduce_sum(tf.multiply(M1, tf.transpose(alpha21, [1, 2, 0] )), 1)
backward_rep = tf.tanh(r1)


output_conc = tf.concat([forward_rep, backward_rep], 1)

print(output_conc)
exit()



#output_conc = tf.nn.dropout(output_conc, 0.7)

logits = tf.matmul(output_conc, W2) + b2
predictions = tf.nn.softmax(logits)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batchY_placeholder))
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


'''
Now I need to prepare the labels as well for training as well as testing 
'''        


#build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)

'''
Things to think about:
1. The training data augmentation is troubling me.
2. Using xavier initializer(or anything else) to initialize weights
3. Do I need to initialize  
'''
x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
        
x_train_left = np.array(x_forward)
x_train_right = np.array(x_backward)
x_test_left = np.array(x_test_f)
x_test_right = np.array(x_test_r)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    

    accuracies = []
    for epoch_idx in range(len(SGD_lrate3)):
        
        # shuffle the training data 
        shuffled_indices = np.random.permutation(np.arange(len(x_forward)))
        shuffled_x_left = x_train_left[shuffled_indices]
        shuffled_x_right = x_train_right[shuffled_indices]

        shuffled_y = y_train[shuffled_indices]
        epoch_loss = 0
        num_batches = 278
        lr = SGD_lrate3[epoch_idx]
        print("New epoch", epoch_idx)
        acc = 0.0
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batchX_left = shuffled_x_left[start_idx:end_idx]
            batchX_right = shuffled_x_right[start_idx:end_idx]
            batchY = shuffled_y[start_idx:end_idx]
            
                       
            _predictions, _accuracy, _total_loss, _train_step = sess.run(
                [ predictions, accuracy, cost, train_step],
                feed_dict={
                    batchX_placeholder: batchX_left,
                    batchX_placeholder_rev: batchX_right,
                    batchY_placeholder: batchY,
                    keep_probability: 0.5,
                    l_rate:lr
                    
                })
            
            epoch_loss += _total_loss
            acc += _accuracy
            
          
        print("epoch loss", epoch_loss)        
        print('Epoch Accuracy', acc/num_batches)




        
        test_predictions, test_accuracy, test_loss  = sess.run(
                [predictions, accuracy, cost],
                feed_dict={
                batchX_placeholder: x_test_left,
                batchX_placeholder_rev: x_test_right,
                batchY_placeholder: y_test,
                keep_probability: 1.0,
                l_rate: 0.001                    
                }
            )

        print('Test Loss: ', test_loss)        
        print('test accuracy :', test_accuracy)
        accuracies.append(test_accuracy)
        target = np.argmax(y_test, 1)
        preds = np.argmax(test_predictions, 1)
        print(preds.shape)
        preds = np.add(preds, 1)
        with open('SDP_att_predictions', 'w') as f:
            for item in preds:
                f.write("%s\n" % item)
        #with open('tf_rnn_cpoy6_copy_targets', 'w') as f:
        #    for item in target:
        #        f.write("%s\n" % item)        
        os.system('python torch_eval.py SDP_att_predictions test_target_modified %d'%epoch_idx)
    print(accuracies)    
            

               


#############################################################################################################
# I have used Max Pooling here F-1 peak was 74.15 in first run. Accuracy peaked at 80 though not at the same time as F-1
# 
'''
I should do something about the variable learning rate and l2 penalty 
Also should I make word2vecs trainable or not?
'''
#############################################################################################################