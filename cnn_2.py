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



batch_size = 50
nb_filter = 1000
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 60
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
filter_shape = [3,420,nb_filter]
W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
output_cnv = tf.nn.conv1d(value=input_conc, filters=W, stride=1, padding='SAME')


h = tf.tanh(tf.nn.bias_add(output_cnv, b), name="Hyperbolic")



        # Max-pooling over the outputs
'''
pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")


pooled_outputs.append(pooled)
'''
#print(output_cnv)
#print(W)


h = tf.expand_dims(h,axis=1)

        # Max-pooling over the outputs

pooled = tf.nn.max_pool(
            h,
            ksize=[1, 1, 97, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

pooled = tf.squeeze(pooled)


##################################
# Define the class matrix shape = (18, )

r = math.sqrt(6.0/(18.0+nb_filter))
#W_classes = np.random.uniform(-r, r, (nb_filter, 18))

W_classes = tf.Variable(np.random.uniform(-r, r, (nb_filter, 18)),dtype=tf.float32)

class_scores = tf.matmul(pooled, W_classes)

n_correct = tf.Variable(0, trainable=True)

for t in xrange(batch_size):
    max_arg = tf.cast(tf.argmax(class_scores[t], 1), tf.int32)
    #true_class = tf.constant(0)
    true_class = tf.cast(tf.argmax(Y[t], 1), tf.int32)
    
    pred_class = tf.Variable(0,trainable=True)
    value = class_scores[t][max_arg]
    tf.cond(value <= 0, lambda: tf.assign(pred_class, 0), lambda: tf.assign(pred_class, max_arg + 1))
    tf.cond(tf.equal(true_class, pred_class), lambda: tf.add(n_correct, 1), lambda: tf.add(n_correct, 0))
    
    #print(value)

accuracy = tf.cast(n_correct, tf.float32)/tf.cast(batch_size, tf.float32)
#n_correct = n_correct

###############################################################################
#   Loss Function

gamma = tf.constant(2.0) 
m_plus = tf.constant(2.5)   
m_minus = tf.constant(0.5)

batch_loss = tf.Variable(0.0, trainable=True)


for t in xrange(batch_size):
    max_arg = tf.cast(tf.argmax(class_scores[t], 1), tf.int32)
    true_class = tf.cast(tf.argmax(Y[t], 1), tf.int32)
    
    top2_val, top2_i = tf.nn.top_k(class_scores[t], 2, sorted=True)  
    
    pred_class = tf.Variable(0, trainable=True)
    true_score = tf.Variable(0.0, trainable=True)
    neg_score = tf.Variable(0.0, trainable=True)

    value = class_scores[t][max_arg]

    tf.cond(value <= 0, lambda: tf.assign(pred_class, 0), lambda: tf.assign(pred_class, max_arg + 1))
    
    tf.cond(tf.equal(true_class, 0), lambda: tf.assign(true_score, 0), lambda: tf.assign(true_score, class_scores[t][true_class-1]))
    
    tf.cond(tf.equal(true_class, 0), lambda: tf.assign(neg_score, value), lambda: tf.cond(tf.equal(true_class, pred_class), 
                lambda: tf.assign(neg_score, top2_val[1]), lambda: tf.assign(neg_score, value)))

    example_loss = tf.Variable(0.0, trainable=True) 
    
    tf.cond(tf.equal(true_class, 0), lambda: tf.assign(example_loss, tf.log(1 + tf.exp(tf.multiply(gamma, m_minus + neg_score)))), 
                    lambda: tf.assign(example_loss, tf.log(1 + tf.exp(tf.multiply(gamma, m_plus - true_score))) + tf.log(1 + tf.exp(tf.multiply(gamma, m_minus + neg_score)))))
    batch_loss = tf.add(batch_loss, example_loss)
    
    #print(neg_score)

batch_loss = batch_loss/batch_size

lambda_t = 0.025
params = tf.trainable_variables()
for theta in params:
    print(theta)

grads = tf.gradients(batch_loss, params)
print(grads)
exit()
train_step = tf.train.GradientDescentOptimizer(learning_rate=lambda_t).minimize(batch_loss)

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
        print('Epoch No ---- ', epoch_idx + 1)

        #sess.eval(print())
        #print(grads)

        for batch_idx in range(num_batches):
            
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batchX = shuffled_X[start_idx:end_idx]
            batchY = shuffled_Y[start_idx:end_idx]
            
            batch_pos1 = shuffled_pos1[start_idx:end_idx]
            batch_pos2 = shuffled_pos2[start_idx:end_idx]
            
                                   
            _total_loss, _train_step,  _accuracy = sess.run(
                    [batch_loss, train_step, accuracy],
                    feed_dict={
                        X: batchX, 
                        X_pos1: batch_pos1,
                        X_pos2: batch_pos2,
                        Y: batchY        
                    }
                )

            #print('batch_loss', _total_loss)
            #print('batch Accuracy', _accuracy)




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