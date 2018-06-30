# coding=utf-8
'''
@author: ***  Chandan Pandey
Using Bidirectional LSTM for relational classification
Will use Glove 100 dim wordvecs    

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
from file_io import get_curr_file_name_no_pfix
import os
import numpy as np
import cPickle
import struct


#############################################
# hyper_param
    
learning_rate = 1.0
num_epochs = 100
truncated_backprop_length = 15  # variable dynamic_unrolling
lstm_size = 100  # same as Embedding size
num_classes = 19
batch_size = 10
# For Drop out
input_p, output_p = 0.3, 0.3
penultimate_p = 0.5

#############################################

#file_train = "sem_train_8000.txt"
#file_test = "sem_test_2717.txt"


file_dict = 'Sem_Eval_dict.pickle'
file_dict_r = 'Sem_dict_r.pickle'
file_embeddings = 'Glove_Embeddings.npy'

path_raw_data = "/home/chandan/Downloads/RE011/ReCly0.11/"
data_path = os.path.join(path_raw_data, 'data')

# loading the files 
with open(os.path.join(data_path, file_dict), 'rb') as file:
    Sem_dict = cPickle.load(file)

with open(os.path.join(data_path, file_dict_r), 'rb') as file:
    Sem_dict_r = cPickle.load(file)

Embeddings = np.load(os.path.join(data_path, file_embeddings))

with open(os.path.join(data_path, 'train_15.pickle'), 'rb') as file:
    train_15_l = cPickle.load(file)

print(Embeddings.shape)
print(len(Sem_dict))
exit()

####################################################################################################
# Placeholders
####################################################################################################

batchX_placeholder = tf.placeholder(tf.int32, [batch_size, None], 'Sentence')
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes], 'label')
backprop_length = tf.placeholder(tf.int32, [batch_size])

###################################################################################################
# Glove Embeddings
###################################################################################################

with tf.variable_scope("embedding"):
    Glove_embedings = tf.get_variable(name="Glove_embedings", shape=Embeddings.shape, 
                        initializer=tf.constant_initializer(Embeddings), trainable=True, dtype='float64')


    input_forward = tf.nn.embedding_lookup(Glove_embedings, batchX_placeholder)

#print (input_forward)
#print (length_seq(input_forward))

#exit()

with tf.variable_scope("Bid_LSTM"):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)

    #lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)
    
    lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
    
    #lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

    rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm, cell_bw=lstm_back,
                                            inputs=input_forward, sequence_length=backprop_length,dtype='float64')

    added_outputs = tf.add(rnn_outputs[0], rnn_outputs[1])
    


# Elementwise sum of the ops check if it is correct 

#######################################################################
# Soft Attention 
#######################################################################
with tf.variable_scope("Soft_Attention"):
    M = tf.tanh(added_outputs)
    #print(M)
    #print(added_outputs)
    #exit()
    attention_w = tf.Variable(np.random.rand(lstm_size, 1),dtype=tf.float64, name='weights')
    
    stacked_w = tf.stack([attention_w for i in xrange(batch_size)])
    alpha = tf.matmul(M, stacked_w)
    
    r = tf.matmul(tf.transpose(added_outputs, perm=[0,2,1]), alpha) 
    #print(r)

    #exit()    
    
    '''
    alpha = tf.matmul(M[0], attention_w)
    alpha = tf.expand_dims(alpha, axis=0)
    
    for i in xrange (1, batch_size):
        alpha_i = tf.matmul(M[i], attention_w)
        alpha_i = tf.expand_dims(alpha_i, axis=0)
        alpha = tf.concat([alpha, alpha_i], axis=0)    
        
    '''

    #alpha = tf.matmul(M[0], tf.transpose(attention_w, perm=[0,1]))
    #print(M[1])
    #print(alpha)
    # Sentence rep -> r = H.alpha.T
    #print (added_outputs[0])
    #print (alpha[0])
    '''
    r = tf.matmul(tf.transpose(added_outputs[0], perm=[1,0]), alpha[0])
    r = tf.expand_dims(r, axis=0)
    #print (r)
    for i in xrange(1, batch_size):
        r_i = tf.matmul(tf.transpose(added_outputs[i], perm=[1,0]), alpha[i])
        r_i = tf.expand_dims(r_i, axis=0)
        r = tf.concat([r, r_i], axis=0)    
    '''
    #print(r)
    r = tf.squeeze(r, axis=2)
    #print(r)
    
    Sentence_rep = tf.tanh(r)
    
    # Do it later 
    #sent_after_dropout = tf.nn.dropout(Sentence_rep, penultimate_p)


with tf.variable_scope("Soft_Max"):
    W2 = tf.Variable(np.random.rand(lstm_size, num_classes),dtype=tf.float64, name='weights')
    b2 = tf.Variable(np.zeros((num_classes)), dtype=tf.float64, name='biases')
    stacked_b2 = tf.stack([b2 for i in xrange(batch_size)])
    
    #print(stacked_b2)
    #print(W2)
    
    logits = tf.matmul(Sentence_rep, W2) + stacked_b2

    predictions = tf.nn.softmax(logits)

#print(logits)
#exit()

l2_lambda = .00001
params = tf.trainable_variables()


#l2_w = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]



L2_loss = 0.0


for p in params:
    if p.name[-4] == 't' or p.name[-4] == 'g':
        L2_loss += l2_lambda * tf.nn.l2_loss(p)
        #print(p)

#exit()        



with tf.variable_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=batchY_placeholder))
    #l2_loss = tf.nn.l2_loss(weights)
    cost = cost + L2_loss
    train_step = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.variable_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(batchY_placeholder,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []

    accuracies = []
    #load the data and  the sequence length
    with open(os.path.join(data_path, 'train_X_wo_bucket.pickle'), 'rb') as file:
        train_x = cPickle.load(file)

    with open(os.path.join(data_path, 'train_Y_wo_bucket.pickle'), 'rb') as file:
        train_y = cPickle.load(file)    
    
    with open(os.path.join(data_path, 'test_X_wo_bucket.pickle'), 'rb') as file:
        test_x = cPickle.load(file)

    with open(os.path.join(data_path, 'test_Y_wo_bucket.pickle'), 'rb') as file:
        test_y = cPickle.load(file)

    with open(os.path.join(data_path, 'train_lengths.pickle'), 'rb') as file:
        train_lengths = cPickle.load(file)

    with open(os.path.join(data_path, 'test_lengths.pickle'), 'rb') as file:
        test_lengths = cPickle.load(file)        

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_y = np.eye(num_classes)[train_y]
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_y = np.eye(num_classes)[test_y]
    train_lengths = np.array(train_lengths)
    test_lengths = np.array(test_lengths)


    for epoch_idx in range(num_epochs):
    
        shuffled_indices = np.random.permutation(np.arange(len(train_y)))
        
        shuffled_train_x = train_x[shuffled_indices]
        shuffled_train_y = train_y[shuffled_indices]
        shuffled_train_lengths = train_lengths[shuffled_indices]
        #print (train_x)
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = int(8000/batch_size)
        print("New epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            batchX_train = list(shuffled_train_x[start_idx:end_idx])
            batchY_train = shuffled_train_y[start_idx:end_idx]
            batch_seq_length = shuffled_train_lengths[start_idx:end_idx]
            #batch_seq_length = np.reshape(batch_seq_length,(batch_size))
            #batch_seq_length = tf.squeeze(batch_seq_length)
            #batch_seq_length = np.array(batch_seq_length)
            #print(batchY_train.shape,batch_seq_length.shape)
            #exit()
            _total_loss,  _predictions, _train_step, _accuracy = sess.run(
                [cost,  predictions, train_step, accuracy],
                feed_dict={
                    batchX_placeholder: batchX_train,
                    batchY_placeholder: batchY_train,
                    backprop_length: batch_seq_length
                    
                })
            #print('batch label unique', np.unique(np.argmax(batchY, 1))) 19 labels fine 

            epoch_loss += _total_loss
            epoch_acc += _accuracy

            loss_list.append(_total_loss)
            if batch_idx%100 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
                print("Step",batch_idx, "Batch accuracy", _accuracy)
        print("epoch loss", epoch_loss)
        print("Epoch Training Accuracy", epoch_acc/num_batches) 

    # Test Accuracy in each epoch
        test_accuracy = 0.0

        

        test_batches = 271
        test_accuracy = 0.0
        total_test_preds = np.zeros(shape=(1, 19))
        last_test = test_y[-1]
        test_y[0] = last_test
        last_test_x = test_x[-1]
        test_x[0] = last_test_x 
        last_test_l = test_lengths[-1]
        test_lengths[0] = last_test_l

        for batch_idx in range(test_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            test_batchX = test_x[start_idx:end_idx]
            test_batchY = test_y[start_idx:end_idx]
            batch_test_lengths = test_lengths[start_idx:end_idx]
            #print(test_batchXt)
            test_predictions, _accuracy  = sess.run(
            [predictions, accuracy],
            feed_dict={
                    batchX_placeholder: test_batchX,
                    batchY_placeholder: test_batchY,
                    backprop_length: batch_test_lengths
                    
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
        y_test_m = test_y[:2710]
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
        
        True_positive = np.zeros(10)
        xdir = np.zeros(9)

        
        row_sum = np.sum(confus_matrix, axis=1)
        column_sum = np.sum(confus_matrix, axis=0)

        

        print(confus_matrix)    
        # calculating the wrong directions 
        for j in xrange(0, 9):
            True_positive[j] = confus_matrix[2*j][2*j] + confus_matrix[2*j+1][2*j+1]
            if True_positive[j] == 0.0:
                True_positive[j] = 1e-6
            xdir[j] = confus_matrix[2*j][2*j+1] + confus_matrix[2*j+1][2*j]    
            
        True_positive[9] = confus_matrix[18][18]    

               
            
        # calculate F-1 score  
        # Precision first 
        Precision_m = np.zeros(10)
        Recall_m = np.zeros(10)
        F1_score = np.zeros(10)
        for i in xrange(0, 9):
            Precision_m[i] = (True_positive[i]) / (column_sum[2*i] + column_sum[2*i + 1] + xdir[i])
            Recall_m[i] = (True_positive[i]) / (row_sum[2*i] + row_sum[2*i + 1])
        
        Precision_m[9] = True_positive[9]/column_sum[18]
        Recall_m[9] = True_positive[9]/row_sum[18]

        Macro_F1 = 0.0
        for i in xrange(0, 10):
            F1_score[i] = (2 * (Precision_m[i] * Recall_m[i])) / (Precision_m[i] + Recall_m[i])
            Macro_F1 += F1_score[i]

        print('Macro F-1 excluding Other', (Macro_F1 - F1_score[9])/9.0)
        print('Macro F-1 including Other', (Macro_F1)/10.0)
                
            
            

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    _current_cell_state_forward = np.zeros((batch_size, state_size))
    _current_cell_state_backward = np.zeros((batch_size, state_size))
    _current_hidden_state_forward = np.zeros((batch_size, state_size))
    _current_hidden_state_backward = np.zeros((batch_size, state_size))

    _current_cell_state_POS_forward = np.zeros((batch_size, pos_rec_1))
    _current_cell_state_POS_backward = np.zeros((batch_size, pos_rec_1))
    _current_hidden_state_POS_forward = np.zeros((batch_size, pos_rec_1))
    _current_hidden_state_POS_backward = np.zeros((batch_size, pos_rec_1))


    accuracies = []
    for epoch_idx in range(num_epochs):
        x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
        POS_X_train_f, POS_X_train_r, Wnet_X_train_f, Wnet_X_train_r, GR_X_train_f,GR_X_train_r = load_POS_GR_Wnet_train(train_b, train_e, valid_b, valid_e, test_b, test_e)
        POS_X_test_f, POS_X_test_r, Wnet_X_test_f, Wnet_X_test_r, GR_X_test_f, GR_X_test_r  =  load_POS_GR_Wnet_test(train_b, train_e, valid_b, valid_e, test_b, test_e)   
    
        x_train_left = np.array(x_forward)
        x_train_right = np.array(x_backward)
        x_test_left = np.array(x_test_f)
        x_test_right = np.array(x_test_r)

         
        # shuffle the training data 
        shuffled_indices = np.random.permutation(np.arange(len(x_forward)))
        shuffled_x_left = x_train_left[shuffled_indices]
        shuffled_x_right = x_train_right[shuffled_indices]
        POS_shuffled_x_f = POS_X_train_f[shuffled_indices]
        POS_shuffled_x_r = POS_X_train_r[shuffled_indices]
        
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
            
                                   
            _total_loss, _train_step, _current_state_forward, _current_state_backward, _current_state_POS_f,_current_state_POS_r, _predictions = sess.run(
                [cost, train_step, state_forward, state_backward, POS_state_f, POS_state_r, predictions],
                feed_dict={
                    batchX_placeholder: batchX_left,
                    batchX_placeholder_rev: batchX_right,
                    batchY_placeholder: batchY,
                    batchX_placeholder_POS: batchX_POS_f,
                    batchX_placeholder_POS_rev: batchX_POS_r,
                    keep_probability: 0.5,
                    cell_state_forward: _current_cell_state_forward,
                    hidden_state_forward: _current_hidden_state_forward,
                    cell_state_backward : _current_cell_state_backward,
                    hidden_state_backward: _current_hidden_state_backward,
                    cell_state_POS_forward: _current_cell_state_POS_forward,
                    hidden_state_POS_forward: _current_hidden_state_POS_forward,
                    cell_state_POS_backward: _current_cell_state_POS_backward,
                    hidden_state_POS_backward: _current_hidden_state_POS_backward
                                

                })
            #print('batch label unique', np.unique(np.argmax(batchY, 1))) 19 labels fine 

            _current_cell_state_forward, _current_hidden_state_forward = _current_state_forward
            _current_cell_state_backward, _current_hidden_state_backward = _current_state_backward
            _current_cell_state_POS_forward, _current_hidden_state_POS_forward = _current_state_POS_f
            _current_cell_state_POS_backward, _current_hidden_state_POS_backward = _current_state_POS_r
            
            epoch_loss += _total_loss

            loss_list.append(_total_loss)

            if batch_idx%10 == 0:
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
            test_batchY = y_test[start_idx:end_idx]
            batchX_POS_test_f = POS_X_test_f[start_idx:end_idx]
            batchX_POS_test_r = POS_X_test_r[start_idx:end_idx]
            
            #print(test)
            test_predictions, _accuracy  = sess.run(
                    [predictions, accuracy],
                    feed_dict={
                        batchX_placeholder: test_batchX_left,
                        batchX_placeholder_rev: test_batchX_right,
                        batchY_placeholder: test_batchY,
                        batchX_placeholder_POS: batchX_POS_f,
                        batchX_placeholder_POS_rev: batchX_POS_r,
                        keep_probability: 1.0,
                        cell_state_forward: _current_cell_state_forward,
                        hidden_state_forward: _current_hidden_state_forward,
                        cell_state_backward : _current_cell_state_backward,
                        hidden_state_backward: _current_hidden_state_backward,
                        cell_state_POS_forward: _current_cell_state_POS_forward,
                        hidden_state_POS_forward: _current_hidden_state_POS_forward,
                        cell_state_POS_backward: _current_cell_state_POS_backward,
                        hidden_state_POS_backward: _current_hidden_state_POS_backward
                    

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
        
        True_positive = np.zeros(10)
        xdir = np.zeros(9)

        
        row_sum = np.sum(confus_matrix, axis=1)
        column_sum = np.sum(confus_matrix, axis=0)

        

        print(confus_matrix)    
        # calculating the wrong directions 
        for j in xrange(0, 9):
            True_positive[j] = confus_matrix[2*j][2*j] + confus_matrix[2*j+1][2*j+1]
            if True_positive[j] == 0.0:
                True_positive[j] = 1e-6
            xdir[j] = confus_matrix[2*j][2*j+1] + confus_matrix[2*j+1][2*j]    
            
        True_positive[9] = confus_matrix[18][18]    

               
            
        # calculate F-1 score  
        # Precision first 
        Precision_m = np.zeros(10)
        Recall_m = np.zeros(10)
        F1_score = np.zeros(10)
        for i in xrange(0, 9):
            Precision_m[i] = (True_positive[i]) / (column_sum[2*i] + column_sum[2*i + 1] + xdir[i])
            Recall_m[i] = (True_positive[i]) / (row_sum[2*i] + row_sum[2*i + 1])
        
        Precision_m[9] = True_positive[9]/column_sum[18]
        Recall_m[9] = True_positive[9]/row_sum[18]

        Macro_F1 = 0.0
        for i in xrange(0, 10):
            F1_score[i] = (2 * (Precision_m[i] * Recall_m[i])) / (Precision_m[i] + Recall_m[i])
            Macro_F1 += F1_score[i]

        print('Macro F-1 excluding Other', (Macro_F1 - F1_score[9])/9.0)
        print('Macro F-1 including Other', (Macro_F1)/10.0)
            
        
    print(accuracies)    
        #_current_cell_state_forward, _current_hidden_state_forward = _current_state_forward
        #_current_cell_state_backward, _current_hidden_state_backward = _current_state_backward
            

               
'''

#############################################################################################################
# I have used Max Pooling here F-1 peak was 80.35 in first run. Accuracy peaked at 80 though not at the same time as F-1
# 
'''
I should do something about the variable learning rate and l2 penalty 
Also should I make word2vecs trainable or not?
'''
#############################################################################################################