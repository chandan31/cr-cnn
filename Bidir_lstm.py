"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789


Code was tested with:
- Python 2.7 & Python 3.6
- Theano 0.9.0 & TensorFlow 1.2.1
- Keras 2.0.5
"""
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import gzip


import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.regularizers import Regularizer
from keras.preprocessing import sequence
from keras import regularizers
import tensorflow as tf 
import numpy as np 
import math



batch_size = 50
nb_filter = 1000
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50
num_classes = 19
lstm_size = 200
print("Load dataset")
f = gzip.open('pkl/sem-relations.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
yTrain, sentenceTrain, positionTrain1, positionTrain2 = data['train_set']
yTest, sentenceTest, positionTest1, positionTest2  = data['test_set']

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
#train_y_cat = np_utils.to_categorical(yTrain, n_out)
max_sentence_len = sentenceTrain.shape[1]


pos_embeddings = np.random.uniform(-0.75, 0.75, (64,position_dims))
train_l = []
test_l = []

for s in sentenceTrain:
    l = 0
    for w in s:
        if w != 0:
            l += 1

    train_l.append(l)


for s in sentenceTest:
    l = 0
    for w in s:
        if w != 0:
            l += 1

    test_l.append(l)


X = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos1 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos2 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
backprop_length = tf.placeholder(tf.int32, [batch_size])

Y = tf.placeholder(tf.int32, [batch_size, n_out])



with tf.variable_scope("Word_embedding"):
    word_embedings = tf.get_variable(name="word_embedings", shape=embeddings.shape,
                        initializer=tf.constant_initializer(embeddings), trainable=True, dtype='float32')


word_inputs = tf.nn.embedding_lookup(word_embedings, X)


with tf.variable_scope("Position_embedding"):
    position_embedings = tf.get_variable(name="pos_embedings", shape=pos_embeddings.shape,
                        initializer=tf.constant_initializer(pos_embeddings), trainable=False, dtype='float32')


pos1_inputs = tf.nn.embedding_lookup(position_embedings, X_pos1)
pos2_inputs = tf.nn.embedding_lookup(position_embedings, X_pos2)

input_conc = concatenate([word_inputs, pos1_inputs, pos2_inputs])

#embedding_drop_ot = tf.nn.dropout(input_conc, 0.6)


with tf.variable_scope("Bid_LSTM"):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)

    #lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)
    
    lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size, state_is_tuple=True)
    
    #lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

    rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm, cell_bw=lstm_back,
                                            inputs=input_conc, sequence_length=backprop_length,dtype='float32')
    added_outputs = tf.add(rnn_outputs[0], rnn_outputs[1])





with tf.variable_scope("Soft_Attention"):
    M = tf.tanh(added_outputs)
    attention_w = tf.Variable(np.random.rand(lstm_size, 1),dtype=tf.float32, name='weights')
    
    stacked_w = tf.stack([attention_w for i in xrange(batch_size)])
    alpha = tf.nn.softmax(tf.matmul(M, stacked_w))
    print(M)
    print(stacked_w)
    print(alpha)
    r = tf.matmul(tf.transpose(added_outputs, perm=[0,2,1]), alpha) 
    r = tf.squeeze(r, axis=2)
    #print(r)
    
    Sentence_rep = tf.tanh(r)



with tf.variable_scope("Soft_Max"):
    W2 = tf.Variable(np.random.rand(lstm_size, num_classes),dtype=tf.float32, name='weights')
    b2 = tf.Variable(np.zeros((num_classes)), dtype=tf.float32, name='biases')
    stacked_b2 = tf.stack([b2 for i in xrange(batch_size)])
    
    #print(stacked_b2)
    #print(W2)
    
    logits = tf.matmul(Sentence_rep, W2) + stacked_b2

    predictions = tf.nn.softmax(logits)


l2_lambda = .00001
params = tf.trainable_variables()

L2_loss = 0.0

'''
for p in params:
    print(p.name)
    if p.name[-4] == 't' or p.name[-4] == 'g':
        
        L2_loss += l2_lambda * tf.nn.l2_loss(p)

'''


with tf.variable_scope("Loss"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    #l2_loss = tf.nn.l2_loss(weights)
    cost = cost + L2_loss
    #train_step = tf.train.AdadeltaOptimizer(learning_rate=1.0).minimize(cost)
with tf.variable_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
train_step = tf.train.AdamOptimizer(learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(cost)

#exit()




#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(cost)



print("Start training")




print(yTrain.shape)
yTrain = np.eye(num_classes)[yTrain]
yTest  = np.eye(num_classes)[yTest]

#last   = yTest[-1]
yTest[0] = yTest[-1]
positionTest1[0] = positionTest1[-1]
positionTest2[0] = positionTest2[-1]
sentenceTest[0] = sentenceTest[-1]

#print(train_Y.shape)
#print(test_Y.shape)
#exit()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []

    print("Start Training ------- ")
    accuracies = []
    lr = 0.025
    train_l = np.array(train_l)
    for epoch_idx in range(nb_epoch):

        shuffled_indices = np.random.permutation(np.arange(sentenceTrain.shape[0]))
        shuffled_X = sentenceTrain[shuffled_indices]
        shuffled_Y = yTrain[shuffled_indices]

        shuffled_pos1 = positionTrain1[shuffled_indices]
        shuffled_pos2 = positionTrain2[shuffled_indices]

        shuffled_train_l = train_l[shuffled_indices]

        num_batches = int(8000/batch_size)
        print(epoch_idx)

        epoch_loss = 0.0
        epoch_acc = 0.0
        lr = lr/(epoch_idx+1)

        for batch_idx in range(num_batches):

            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            #print(start_idx, end_idx)    

            batchX = shuffled_X[start_idx:end_idx]
            batchY = shuffled_Y[start_idx:end_idx]
            #print(batchX)
            #print(batchY)
            batchX_l = shuffled_train_l[start_idx:end_idx]
            batch_pos1 = shuffled_pos1[start_idx:end_idx]
            batch_pos2 = shuffled_pos2[start_idx:end_idx]


            _total_loss, _train_step,  _accuracy, _predictions = sess.run(
                    [cost, train_step, accuracy, predictions],
                    feed_dict={
                        X: batchX,
                        X_pos1: batch_pos1,
                        X_pos2: batch_pos2,
                        Y: batchY,
                        backprop_length:batchX_l
                    }
                )

            #print('batch_accuracy', _accuracy)  
            epoch_loss += _total_loss
            epoch_acc += _accuracy
            loss_list.append(_total_loss)


        print("epoch loss", epoch_loss)
        print("Epoch Accuracy", epoch_acc/num_batches)

        # Testing
        
        test_acc = 0.0
        test_loss = 0.0

        num_t_batches = int(2710/batch_size)

        for batch_idx in range(num_t_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            #print(start_idx, end_idx)    
            batchX_test = sentenceTest[start_idx:end_idx]
            batchY_test = yTest[start_idx:end_idx]


            #print(batchX)
            #print(batchY)

            batch_t_pos1 = positionTest1[start_idx:end_idx]
            batch_t_pos2 = positionTest2[start_idx:end_idx]


            _test_loss, test_accuracy, _predictions = sess.run(
                    [cost, accuracy, predictions],
                    feed_dict={
                        X: batchX_test,
                        X_pos1: batch_t_pos1,
                        X_pos2: batch_t_pos2,
                        Y: batchY_test,
                        l_rate: 0.001
                    }
                )

            #print('batch_accuracy', _accuracy)  
            test_loss += _test_loss
            test_acc += test_accuracy
            loss_list.append(_total_loss)


        print("epoch test loss", test_loss)
        print("Epoch Test Accuracy", test_acc/num_t_batches)
        accuracies.append(test_acc/num_t_batches)
    print('Max Test Acc', max(accuracies))
    