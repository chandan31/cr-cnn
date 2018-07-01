"""
This is a CNN for relation classification within a sentence. The architecture is based on:

Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou and Jun Zhao, 2014, Relation Classification via Convolutional Deep Neural Network

Performance (without hyperparameter optimization):
Accuracy: 0.7943
Macro-Averaged F1 (without Other relation):  0.7612

Performance Zeng et al.
Macro-Averaged F1 (without Other relation): 0.789

@Chandan Accuracy = 79.70
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
import tensorflow as tf
import numpy as np
import math



batch_size = 50
nb_filter = 900
filter_length = 3
hidden_dims = 100
nb_epoch = 500
position_dims = 50
num_classes = 19

print("Load dataset")
f = gzip.open('pkl/sem-relations3.pkl.gz', 'rb')
data = pkl.load(f)
f.close()

embeddings = data['wordEmbeddings']
yTrain, sentenceTrain, positionTrain1, positionTrain2, e1_pos, e2_pos, neg_class = data['train_set']

yTest, sentenceTest, positionTest1, positionTest2, e1_pos_test, e2_pos_test, neg_class_t  = data['test_set']

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

#######################################
#   Calculating the negative class indices 18 dimension vector
#############################################





#print(e1_pos)

n_out = max(yTrain)+1

max_sentence_len = sentenceTrain.shape[1]


pos_embeddings = np.random.uniform(-0.65, 0.65, (64,position_dims))
#pos_embeddings = np.random.rand(64,position_dims)

X = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos1 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos2 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
neg_idx = tf.placeholder(tf.int32, [batch_size, 18])

#e1_placeholder = tf.placeholder(tf.int32, [batch_size])
#e2_placeholder = tf.placeholder(tf.int32, [batch_size])



Y = tf.placeholder(tf.int32, [batch_size, n_out])
l_rate = tf.placeholder(tf.float32)


with tf.variable_scope("Word_embedding"):
    word_embedings = tf.get_variable(name="word_embeding", shape=embeddings.shape,
                        initializer=tf.constant_initializer(embeddings), trainable=False, dtype='float32')


word_inputs = tf.nn.embedding_lookup(word_embedings, X)

#print(e1)
#print(word_inputs)
###################################################################
#   Apply Primary attention here 
###################################################################
'''
e1 = tf.nn.embedding_lookup(word_embedings, e1_placeholder)
e2 = tf.nn.embedding_lookup(word_embedings, e2_placeholder)


e1 = tf.expand_dims(e1, axis=2)
e2 = tf.expand_dims(e2, axis=2)


e1_att =  tf.matmul(word_inputs, e1)
e2_att = tf.matmul(word_inputs, e2)

e1_att = tf.squeeze(e1_att)
e2_att = tf.squeeze(e2_att)

e1_score = tf.nn.softmax(e1_att)
e2_score = tf.nn.softmax(e2_att)

avg_scores = tf.add(e1_score, e2_score)
avg_scores = tf.div(avg_scores, 2.0)

stacked_sc = tf.stack([avg_scores for i in xrange(300)], axis=2)
print (stacked_sc)
input_words = tf.multiply(word_inputs, stacked_sc)
'''
###################################################################
###################################################################
#print(avg_scores)
#print(input_words)
#exit()

with tf.variable_scope("Position_embedding"):
    position_embedings = tf.get_variable(name="pos_embedings", shape=pos_embeddings.shape,
                        initializer=tf.constant_initializer(pos_embeddings), trainable=False, dtype='float32')


pos1_inputs = tf.nn.embedding_lookup(position_embedings, X_pos1)
pos2_inputs = tf.nn.embedding_lookup(position_embedings, X_pos2)


input_conc = concatenate([word_inputs, pos1_inputs, pos2_inputs])


expanded_in = tf.expand_dims(input_conc, axis=3)


filter_shape = [3,400,nb_filter]
#W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
W = tf.get_variable("F_Weight", shape=[3, 400, nb_filter],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b = tf.get_variable(name="F_bias", shape=[nb_filter], initializer=tf.zeros_initializer())

output_cnv = tf.nn.conv1d(value=input_conc, filters=W, stride=1, padding='SAME')


h = tf.tanh(tf.nn.bias_add(output_cnv, b), name="Hyperbolic")

h = tf.expand_dims(h,axis=1)

        # Max-pooling over the outputs

pooled = tf.nn.max_pool(
            h,
            ksize=[1, 1, 97, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

pooled = tf.squeeze(pooled)

#pooled_outputs.append(pooled)

#print(output_cnv)
#print(W)

drop_ot = tf.nn.dropout(pooled, 0.6)


r = math.sqrt(6.0/(18.0+nb_filter))

#print(r)
#W_classes = np.random.uniform(-r, r, (nb_filter, 18))
W_C = tf.random_uniform(shape=[nb_filter, 18], minval=-r, maxval=r, dtype=tf.float32)

W_classes = tf.get_variable("my_int_variable", dtype=tf.float32, 
                    initializer=W_C)

#W_classes = tf.Variable(np.random.uniform(-r, r, (nb_filter, 18)),dtype=tf.float32)

class_scores = tf.matmul(pooled, W_classes)



###################################################################################################
#   Prediction and Ranking layer
####################################################################################################

#   Prediction: 
# Append -max value in the end of class score

max_score = tf.reduce_max(class_scores, axis=1)

max_score = tf.negative(max_score)


max_score = tf.reshape(max_score,shape=[batch_size,1])

class_scores_m = tf.concat([class_scores,max_score], axis=1)

###
# argmax will be the prediction now
#   If all scores are negative then -max will be the argmax value which is value at other class index(18)
#########################################
correct_pred = tf.equal(tf.argmax(class_scores_m, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#print(correct_pred) Doubts 
#print(accuracy)


predictions = tf.nn.softmax(class_scores_m)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))

p = tf.trainable_variables()
'''
L2_loss = 0.0
for p_ in p:
    print(p_.name[-3])
    if p_.name[-3] == 't': # or p_.name[-3] == 'g':
        L2_loss += tf.nn.l2_loss(p_)
'''

#exit()

train_step = tf.train.AdamOptimizer(learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(cost)

#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(cost)


print(yTrain.shape)
yTrain = np.eye(num_classes)[yTrain]
yTest  = np.eye(num_classes)[yTest]

#last   = yTest[-1]
yTest[0] = yTest[-1]
positionTest1[0] = positionTest1[-1]
positionTest2[0] = positionTest2[-1]
sentenceTest[0] = sentenceTest[-1]
neg_class[0] = neg_class[-1]
#print(train_Y.shape)
#print(test_Y.shape)
#exit()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []

    print("Start Training ------- ")
    accuracies = []
    lr = 0.025
    for epoch_idx in range(nb_epoch):


        shuffled_indices = np.random.permutation(np.arange(sentenceTrain.shape[0]))
        shuffled_X = sentenceTrain[shuffled_indices]
        shuffled_Y = yTrain[shuffled_indices]

        shuffled_pos1 = positionTrain1[shuffled_indices]
        shuffled_pos2 = positionTrain2[shuffled_indices]

        shuffled_e1 = e1_pos[shuffled_indices]
        shuffled_e2 = e2_pos[shuffled_indices]
        shuffled_neg_idx = neg_class[shuffled_indices]

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

            batch_pos1 = shuffled_pos1[start_idx:end_idx]
            batch_pos2 = shuffled_pos2[start_idx:end_idx]

            batch_e1 = shuffled_e1[start_idx:end_idx]
            batch_e2 = shuffled_e2[start_idx:end_idx]

            batch_neg_idx = shuffled_neg_idx[start_idx:end_idx]

            _total_loss, _train_step,  _accuracy = sess.run(
                    [cost, train_step, accuracy],
                    feed_dict={
                        X: batchX,
                        X_pos1: batch_pos1,
                        X_pos2: batch_pos2,
                        neg_idx:batch_neg_idx,
                        Y: batchY,
                        l_rate: lr
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
            batch_e1_t = e1_pos_test[start_idx:end_idx]
            batch_e2_t = e2_pos_test[start_idx:end_idx]

            batch_t_pos1 = positionTest1[start_idx:end_idx]
            batch_t_pos2 = positionTest2[start_idx:end_idx]
            t_batch_neg_idx = neg_class_t[start_idx:end_idx]

            _test_loss, test_accuracy = sess.run(
                    [cost, accuracy],
                    feed_dict={
                        X: batchX_test,
                        X_pos1: batch_t_pos1,
                        X_pos2: batch_t_pos2,
                        neg_idx:t_batch_neg_idx,
                        Y: batchY_test,
                        l_rate: lr
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

