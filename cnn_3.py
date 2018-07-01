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
import tensorflow as tf 
import numpy as np 
import math



batch_size = 20
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

#print("sentenceTrain: ", sentenceTrain.shape)
#print("positionTrain1: ", positionTrain1.shape)
#print("yTrain: ", yTrain.shape)




#print("sentenceTest: ", sentenceTest.shape)
#print("positionTest1: ", positionTest1.shape)
#print("yTest: ", yTest.shape)



#print("Embeddings: ",embeddings.shape)

#print(max_sentence_len)
#exit()
'''
def Accuracy(scores, Y):
    for t in xrange(batch_size):
        pred_class = tf.argmax(scores[t])
'''

pos_embeddings = np.random.rand(64,position_dims)

X = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos1 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
X_pos2 = tf.placeholder(tf.int32, [batch_size, max_sentence_len])
Y = tf.placeholder(tf.int32, [batch_size, n_out])

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

#print(output)
#exit()
expanded_in = tf.expand_dims(input_conc, axis=3)
#print(expanded_in)
word_dims = 300 + 2*position_dims
filter_shape = [3,word_dims,nb_filter]
W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

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


learning_rate = 0.1

W2 = tf.Variable(np.random.rand(nb_filter, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

params = tf.trainable_variables()



logits = tf.matmul(pooled, W2) + b2
predictions = tf.nn.softmax(logits)

# define op to calculate F-1 score on test data 

correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.025).minimize(cost)



##################################
# Define the class matrix shape = (18, )


#lambda_t = 0.025
#train_step = tf.train.GradientDescentOptimizer(learning_rate=lambda_t).minimize(batch_loss)
########################
###     L2 Regularizer

'''
Beta = tf.constant(0.001)

params = tf.trainable_variables()

print('Hi')

for theta in params:
    print(theta.name)


'''
print(yTrain.shape)
yTrain = np.eye(num_classes)[yTrain]
yTest  = np.eye(num_classes)[yTest]

#print(train_Y.shape)
#print(test_Y.shape)
#exit()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loss_list = []
    
    print("Start Training ------- ")    
    accuracies = []
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
                        Y: batchY        
                    }
                )

            #print('batch_accuracy', _accuracy)  
            epoch_loss += _total_loss
            epoch_acc += _accuracy
            loss_list.append(_total_loss)

            
        print("epoch loss", epoch_loss)
        print("Epoch Accuracy", epoch_acc/num_batches)              



'''

exit()

output = Dropout(0.25)(output)
output = Dense(n_out, activation='softmax')(output)

model = Model(inputs=[X, X_pos1, X_pos2], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
model.summary()

print("Start training")

max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
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

for epoch in range(nb_epoch):       
    model.fit([sentenceTrain, positionTrain1, positionTrain2], yTrain, batch_size=batch_size, verbose=True,epochs=1)   
    pred_test = predict_classes(model.predict([sentenceTest, positionTest1, positionTest2], verbose=False))
    
    dctLabels = np.sum(pred_test)
    totalDCTLabels = np.sum(yTest)
   
    acc =  np.sum(pred_test == yTest) / float(len(yTest))
    max_acc = max(max_acc, acc)
    print("Accuracy: %.4f (max: %.4f)" % (acc, max_acc))

    f1Sum = 0
    f1Count = 0
    for targetLabel in range(1, max(yTest)):        
        prec = getPrecision(pred_test, yTest, targetLabel)
        recall = getPrecision(yTest, pred_test, targetLabel)
        f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
        f1Sum += f1
        f1Count +=1    
        
        
    macroF1 = f1Sum / float(f1Count)    
    max_f1 = max(max_f1, macroF1)
    print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))
'''
