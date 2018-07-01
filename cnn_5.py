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

X = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos1 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos2 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])

Y = tf.placeholder(tf.int32, [batch_size, n_out])
l_rate = tf.placeholder(tf.float32)
#Keras words_input = Input(shape=(max_sentence_len,), dtype='int32', name='words_input')


with tf.variable_scope("Word_embedding"):
    word_embedings = tf.get_variable(name="word_embedings", shape=embeddings.shape,
                        initializer=tf.constant_initializer(embeddings), trainable=True, dtype='float32')


word_inputs = tf.nn.embedding_lookup(word_embedings, X)


#Keras words = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=True)(words_input)

with tf.variable_scope("Position_embedding"):
    position_embedings = tf.get_variable(name="pos_embedings", shape=pos_embeddings.shape,
                        initializer=tf.constant_initializer(pos_embeddings), trainable=False, dtype='float32')


pos1_inputs = tf.nn.embedding_lookup(position_embedings, X_pos1)
pos2_inputs = tf.nn.embedding_lookup(position_embedings, X_pos2)




input_conc = concatenate([word_inputs, pos1_inputs, pos2_inputs])

expanded_in = tf.expand_dims(input_conc, axis=3)
#print(expanded_in)
filter_shape = [3,400,nb_filter]

W = tf.get_variable("W", shape=filter_shape,
           initializer=tf.contrib.layers.xavier_initializer())

#W = tf.Variable(tf.random_uniform(filter_shape, stddev=0.1), name="W")
#bias1 = np.full((nb_filter), 0.1)

b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
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

w2 = np.random.rand(nb_filter, num_classes)
b2 = np.full((num_classes), 0.1)

W2 = tf.get_variable(name="W2", shape=w2.shape, initializer=tf.constant_initializer(w2), trainable=True, dtype='float32')
B2 = tf.get_variable(name="B2", shape=b2.shape, initializer=tf.constant_initializer(b2), trainable=True, dtype='float32')

params = tf.trainable_variables()



logits = tf.matmul(drop_ot, W2) + B2
predictions = tf.nn.softmax(logits)

# define op to calculate F-1 score on test data 

correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

p = tf.trainable_variables()
#L2_loss = 0.0
for p in params:
    print(p)
L2_loss = tf.nn.l2_loss(p[-2])



#l2_loss = 

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predictions))
cost = cost + (0.01)*L2_loss

train_step = tf.train.AdamOptimizer(learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(cost)

#train_step = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(cost)



print("Start training")

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    #print('1. Pred Test shape', pred_test.shape)
    #print('1. Pred Test', pred_test)
    #print('1. ytest shape', yTest.shape)
    #print('1. yTest', yTest)
    #print('target Label', targetLabel.shape)
    #exit()
    
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount

def predict_classes(prediction):
 return prediction.argmax(axis=-1)



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
    for epoch_idx in range(nb_epoch):

        shuffled_indices = np.random.permutation(np.arange(sentenceTrain.shape[0]))
        shuffled_X = sentenceTrain[shuffled_indices]
        shuffled_Y = yTrain[shuffled_indices]

        shuffled_pos1 = positionTrain1[shuffled_indices]
        shuffled_pos2 = positionTrain2[shuffled_indices]

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


            _total_loss, _train_step,  _accuracy, _predictions = sess.run(
                    [cost, train_step, accuracy, predictions],
                    feed_dict={
                        X: batchX,
                        X_pos1: batch_pos1,
                        X_pos2: batch_pos2,
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

            batch_t_pos1 = positionTest1[start_idx:end_idx]
            batch_t_pos2 = positionTest2[start_idx:end_idx]


            _test_loss, test_accuracy, _predictions = sess.run(
                    [cost, accuracy, predictions],
                    feed_dict={
                        X: batchX_test,
                        X_pos1: batch_t_pos1,
                        X_pos2: batch_t_pos2,
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

