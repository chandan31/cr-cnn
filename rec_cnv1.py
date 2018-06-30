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

SGD_lrate = [0.55, 0.54, 0.536, 0.53, 0.525, 0.521, 0.53, 0.516, 0.51, 0.50, 0.48, 0.477, 0.460, 0.445, 0.433, 0.428, 0.420, 0.39, 0.36, 0.348, 0.32, 0.31, 
                0.3, 0.287, 0.27, 0.243, 0.2187, 0.19683, 0.177147, 0.3, 0.27, 0.21, 0.17,
                0.159432, 0.143489, 0.12914, 0.116226, 0.104604, 0.0941432, 0.0847288, 0.076256, 0.0686304, 0.0617673, 0.0555906, 
                0.0500315, 0.0450284, 0.0405255, 0.036473, 0.0328257, 0.0295431, 0.0265888, 0.0239299, 0.0215369, 0.0193832, 
                0.0174449, 0.0157004, 0.0141304, 0.0127173, 0.0114456, 0.010301, 0.00927094, 0.00834385, 0.00750946, 0.00675852, 0.00608266, 
                0.0054744, 0.00492696, 0.00443426, 0.0039, 0.00343426, 0.00253426]



batchX_placeholder = tf.placeholder(tf.int32, [None, truncated_backprop_length])
batchX_placeholder_rev = tf.placeholder(tf.int32, [None, truncated_backprop_length])

batchY_placeholder = tf.placeholder(tf.int32, [None, num_classes]) # perhaps I am right ??
keep_probability = tf.placeholder(tf.float32)
l_rate = tf.placeholder(tf.float32)

Wordvec_embedings = tf.get_variable(name="Wordvec_embediings", shape=Embeddings.shape, initializer=tf.constant_initializer(Embeddings), trainable=True)



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

for i in xrange(0, truncated_backprop_length):
    outputs_forward[i] = tf.expand_dims(outputs_forward[i], axis=1)
    outputs_backward[i] = tf.expand_dims(outputs_backward[i], axis=1)

forward_tensor_concat = tf.concat([outputs_forward[0], outputs_forward[1]], 1)
backward_tensor_concat = tf.concat([outputs_backward[0], outputs_backward[1]], 1)

for i in xrange(2, truncated_backprop_length):
    forward_tensor_concat = tf.concat([forward_tensor_concat, outputs_forward[i]], 1)
    backward_tensor_concat = tf.concat([backward_tensor_concat, outputs_backward[i]], 1)

nb_filter1 = 400 
nb_filter2 = 400

W2 = tf.get_variable("Filter_SDP", shape=[5, 150, nb_filter1],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b2 = tf.get_variable(name="Bias_SDP", shape=[nb_filter1], initializer=tf.zeros_initializer())

output_cnv1 = tf.nn.conv1d(value=forward_tensor_concat, filters=W2, stride=1, padding='SAME')


h1 = tf.tanh(tf.nn.bias_add(output_cnv1, b2), name="Hyperbolic1")

h1 = tf.expand_dims(h1, axis=1)

        # Max-pooling over the outputs

pooled1 = tf.nn.max_pool(
            h1,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

pooled1 = tf.squeeze(tf.squeeze(pooled1, 1), 1)


W3 = tf.get_variable("Filter_SDP2", shape=[5, 150, nb_filter1],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b3 = tf.get_variable(name="Bias_SDP2", shape=[nb_filter2], initializer=tf.zeros_initializer())

output_cnv2 = tf.nn.conv1d(value=backward_tensor_concat, filters=W3, stride=1, padding='SAME')


h2 = tf.tanh(tf.nn.bias_add(output_cnv2, b3), name="Hyperbolic1")

h2 = tf.expand_dims(h2, axis=1)

        # Max-pooling over the outputs

pooled2 = tf.nn.max_pool(
            h2,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

pooled2 = tf.squeeze(tf.squeeze(pooled2, 1), 1)

print(pooled2)

W4 = tf.get_variable("W_soft", shape=[800, num_classes],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b4 = tf.get_variable(name="Bias_Softmax", shape=[num_classes], initializer=tf.zeros_initializer())


output_conc = tf.concat([pooled1, pooled2], 1)
output_conc = tf.nn.dropout(output_conc, 0.7)

logits = tf.matmul(output_conc, W4) + b4
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


    y_test = np.zeros(shape=(len(labels_test), 19))
    for i in xrange(0, len(labels_test)):
        one_hot = np.zeros(19)
        label = labels_test[i]
        one_hot[label] = 1
        y_test[i] = one_hot


    return batchX_train_forward, batchX_train_reverse, y_train, batchX_test_forward, batchX_test_reverse, y_test 

 

x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
        
x_train_left = np.array(x_forward)
x_train_right = np.array(x_backward)
x_test_left = np.array(x_test_f)
x_test_right = np.array(x_test_r)



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    

    accuracies = []
    for epoch_idx in range(len(SGD_lrate)):
        
        # shuffle the training data 
        shuffled_indices = np.random.permutation(np.arange(len(x_forward)))
        shuffled_x_left = x_train_left[shuffled_indices]
        shuffled_x_right = x_train_right[shuffled_indices]

        shuffled_y = y_train[shuffled_indices]
        epoch_loss = 0
        num_batches = 278
        lr = SGD_lrate[epoch_idx]
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
        with open('rec_cnv_predictions', 'w') as f:
            for item in preds:
                f.write("%s\n" % item)
        #with open('tf_rnn_cpoy6_copy_targets', 'w') as f:
        #    for item in target:
        #        f.write("%s\n" % item)        
        os.system('python torch_eval.py rec_cnv_predictions test_target_modified')
    print(accuracies)    
            
