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
#print 'Hello! Anybody in there?'
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


wn_embeddings = np.zeros(shape=(45, 50))
pos_embeddings = np.zeros(shape=(16, 50))
gr_embeddings = np.zeros(shape=(20,50))

wn_dic['paddd'] = 0
pos_dic['paddd'] = 0
gr_dic['paddd'] = 0


##############################
'''
Model: LSTM with first  layer of size 128  
'''

num_epochs = 100
truncated_backprop_length = 12
state_size = 200
num_classes = 19
batch_size = 50







batchX_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])
batchX_placeholder_rev = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

batchY_placeholder = tf.placeholder(tf.int32, [batch_size, num_classes]) # perhaps I am right ??
keep_probability = tf.placeholder(tf.float32)
l_rate = tf.placeholder(tf.float32)




# Define Embedding layer 
#embedding = init_weight([FLAGS.vocab_size, FLAGS.embedding_size], "Embedding")
Wordvec_embedings = tf.get_variable(name="Wordvec_embediings", shape=Embeddings.shape, initializer=tf.constant_initializer(Embeddings), trainable=True)




# softmax weights since we are working on concatenated state hence 2*state_size dimension 

W2 = tf.Variable(np.random.rand(100, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)


########################################################################################################
# Inputs after Embedding Lookup 
########################################################################################################
input_forward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder)
input_backward = tf.nn.embedding_lookup(Wordvec_embedings, batchX_placeholder_rev)



Embedding_dropout_left = tf.nn.dropout(input_forward, keep_probability)
Embedding_dropout_right = tf.nn.dropout(input_backward, keep_probability) 

#sliced = tf.slice(input_forward, [0, 0, 0], [batch_size, truncated_backprop_length - 1, 200] )
#print (sliced)

nb_filter1 = 200
nb_filter = 100 


W22 = tf.get_variable("Filter_SDP", shape=[5, 200, nb_filter1],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b22 = tf.get_variable(name="Bias_SDP", shape=[nb_filter1], initializer=tf.zeros_initializer())

W23 = tf.get_variable("Filter_SDP2", shape=[3, 200, nb_filter],
           initializer=tf.contrib.layers.xavier_initializer())

#b = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b")
b23 = tf.get_variable(name="Bias_SDP2", shape=[nb_filter], initializer=tf.zeros_initializer())

output_cnv22 = tf.nn.conv1d(value=Embedding_dropout_left, filters=W22, stride=1, padding='SAME')

output_cnv23 = tf.nn.conv1d(value=Embedding_dropout_left, filters=W23, stride=1, padding='SAME')

h22 = tf.tanh(tf.nn.bias_add(output_cnv21, b22), name="Hyperbolic22")

h22 = tf.expand_dims(h22, axis=1)


h23 = tf.tanh(tf.nn.bias_add(output_cnv23, b23), name="Hyperbolic23")

h23 = tf.expand_dims(h23, axis=1)

pooled22 = tf.nn.max_pool(
            h22,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool22")

pooled22 = tf.squeeze(tf.squeeze(pooled22, 1), 1)

pooled23 = tf.nn.max_pool(
            h23,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool23")

pooled23 = tf.squeeze(tf.squeeze(pooled23, 1), 1)



output_cnv24 = tf.nn.conv1d(value=Embedding_dropout_right, filters=W22, stride=1, padding='SAME')
output_cnv25 = tf.nn.conv1d(value=Embedding_dropout_right, filters=W23, stride=1, padding='SAME')

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










nb_filter = 500

expanded_in = tf.expand_dims(input_forward, axis=3)
#print(expanded_in)
filter_shape = [3, 200, nb_filter]

W1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W1")

b1 = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b1")
output_cnv1 = tf.nn.conv1d(value=input_forward, filters=W1, stride=1, padding='SAME')


h1 = tf.tanh(tf.nn.bias_add(output_cnv1, b1), name="Hyperbolic")

h1 = tf.expand_dims(h1,axis=1)

        # Max-pooling over the outputs

pooled1 = tf.nn.max_pool(
            h1,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")

pooled1 = tf.squeeze(pooled1)

print(pooled1)

expanded_in = tf.expand_dims(input_backward, axis=3)
#print(expanded_in)
filter_shape = [3,200,nb_filter]

W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")

b2 = tf.Variable(tf.constant(0.1, shape=[nb_filter]), name="b2")
output_cnv2 = tf.nn.conv1d(value=input_backward, filters=W2, stride=1, padding='SAME')


h2 = tf.tanh(tf.nn.bias_add(output_cnv2, b2), name="Hyperbolic")

h2 = tf.expand_dims(h2, axis=1)

        # Max-pooling over the outputs

pooled2 = tf.nn.max_pool(
            h2,
            ksize=[1, 1, 12, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool1")

pooled2 = tf.squeeze(pooled2)



pooled = tf.concat([pooled1, pooled2], 1)
#print(pooled)


W2 = tf.Variable(np.random.rand(1000, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)



'''
train_v = tf.trainable_variables()

regularized = []
w = 6
lambda_l2 = 1e-3

while w <23:
    regularized.append(train_v[w])
    w = w + 2
regularized_loss = 0.0
for w in regularized:
    #print(w)
    regularized_loss = regularized_loss + lambda_l2 * tf.nn.l2_loss(w)
'''

#print(regularized_loss)
#print(after_dropout)

logits = tf.matmul(pooled, W2) + b2
print(logits)

predictions = tf.nn.softmax(logits)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=batchY_placeholder))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# define op to calculate F-1 score on test data 

correct_pred = tf.equal(tf.argmax(predictions,1), tf.argmax(batchY_placeholder,1))
predicted_labels = tf.argmax(predictions,1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


#####################################################################
'''
We will prepare the data here 
'''


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



##########################################################################################################################################33
#       For Precision calculation
###################################################################################################33


max_prec, max_rec, max_acc, max_f1 = 0,0,0,0

def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    
    targetLabelCount = 0
    correctTargetLabelCount = 0
    
    for idx in range(pred_test.shape[0]):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    
    if correctTargetLabelCount == 0:
        return 0
    
    return float(correctTargetLabelCount) / targetLabelCount




with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    
    loss_list = []
    

    accuracies = []
    for epoch_idx in range(num_epochs):
        x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
        
        x_train_left = np.array(x_forward)
        x_train_right = np.array(x_backward)
        x_test_left = np.array(x_test_f)
        x_test_right = np.array(x_test_r)

         
        # shuffle the training data 
        shuffled_indices = np.random.permutation(np.arange(len(x_forward)))
        shuffled_x_left = x_train_left[shuffled_indices]
        shuffled_x_right = x_train_right[shuffled_indices]
        


        shuffled_y = y_train[shuffled_indices]
        epoch_loss = 0
        num_batches = 278

        print("New epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batchX_left = shuffled_x_left[start_idx:end_idx]
            batchX_right = shuffled_x_right[start_idx:end_idx]
            batchY = shuffled_y[start_idx:end_idx]
          
                                   
            _total_loss, _train_step, _predictions = sess.run(
                [cost, train_step, predictions],
                feed_dict={
                    batchX_placeholder: batchX_left,
                    batchX_placeholder_rev: batchX_right,
                    batchY_placeholder: batchY,
                    keep_probability: 0.5,
                    l_rate: 0.002                    

                })
            #print('batch label unique', np.unique(np.argmax(batchY, 1))) 19 labels fine 

            
            epoch_loss += _total_loss

            loss_list.append(_total_loss)

            if batch_idx%10 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
        print("epoch loss", epoch_loss)        





        test_batches = 54
        test_accuracy = 0
        total_test_preds = np.zeros(shape=(1, 19))
        
        x_test_left[0] = x_test_left[-1]
        x_test_right[0] = x_test_right[-1]

        last_test = y_test[-1]
        y_test[0] = last_test
        p_labels = np.array([])
        for batch_idx in range(test_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            test_batchX_left = x_test_left[start_idx:end_idx]
            test_batchX_right = x_test_right[start_idx:end_idx]
            test_batchY = y_test[start_idx:end_idx]
            
            #print(test)
            test_predictions, _accuracy, _correct_pred, _predicted_labels  = sess.run(
                    [predictions, accuracy, correct_pred, predicted_labels],
                    feed_dict={
                        batchX_placeholder: test_batchX_left,
                        batchX_placeholder_rev: test_batchX_right,
                        batchY_placeholder: test_batchY,
                        keep_probability: 1.0,
                        l_rate: 0.002

                    })

            #total_test_preds.append(test_predictions)
            test_predictions = np.array(test_predictions)
            #print('test label unique', np.unique(np.argmax(test_batchY, 1)))
            #print(test_predictions.shape)
            total_test_preds = np.concatenate((total_test_preds, test_predictions) , axis=0)
            #print(total_test_preds.shape)
            
            test_accuracy +=  _accuracy
            
            p_labels = np.concatenate((p_labels,_predicted_labels), axis=0)
            #p_labels.append(_predicted_labels)
            

        
        print('test accuracy :', test_accuracy/test_batches)
        accuracies.append(test_accuracy/test_batches)
        
        #print(total_test_preds)
        # So I have the Predictions on all test data. Now I need to calcultae classwise TP and TN FN etc 
        # true positives 

        total_test_preds = np.array(total_test_preds)
        
        #print('Shape of Predcited labels', p_labels)
        #exit()
        y_test_m = y_test[:2700]
        
        total_test_preds_m = total_test_preds[1:]
        
        yTest = np.argmax(y_test_m,1)

        f1Sum = 0
        f1Count = 0
        
        #print(yTest)
        #print(p_labels)

        for targetLabel in range(0, 18):        
            prec = getPrecision(p_labels, yTest, targetLabel)
            recall = getPrecision(yTest, p_labels, targetLabel)
            #print('Target Label', targetLabel)
            #print('Precision', prec)
            #print('Recall', recall)
            f1 = 0 if (prec+recall) == 0 else 2*prec*recall/(prec+recall)
            f1Sum += f1
            f1Count +=1    
        
        
        macroF1 = f1Sum / float(f1Count)    
        max_f1 = max(max_f1, macroF1)
        print("Non-other Macro-Averaged F1: %.4f (max: %.4f)\n" % (macroF1, max_f1))
        
        
        c1 = np.argmax(y_test_m, 1)
        c2 = np.argmax(total_test_preds_m, 1)
        
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
            

               


