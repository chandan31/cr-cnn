from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
from sklearn.metrics import confusion_matrix
import gzip
import os
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl 
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
pos_rec_1 = 25
pos_rec_2 = 50
pos_rec_3 = 50
pos_rec_4 = 50

wn_self  = 50
wn_rec_1 = 50
wn_rec_2 = 50
wn_rec_3 = 50
wn_rec_4 = 50

gr_self  = 50
gr_rec_1 = 25
gr_rec_2 = 50
gr_rec_3 = 50
gr_rec_4 = 50
truncated_backprop_length = 12

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

    
file_train = "sem_train_8000.txt"
file_test = "sem_test_2717.txt"


file_dict = 'Sem_Eval_dict.pickle'
file_dict_r = 'Sem_dict_r.pickle'
file_embeddings = 'Glove_Embeddings.npy'

path_raw_data1 = "/home/chandan/Downloads/RE011/ReCly0.11/"
data_path = os.path.join(path_raw_data1, 'data')

# loading the files 
with open(os.path.join(data_path, file_dict), 'rb') as file:
    vocab_dic = cPickle.load(file)

with open(os.path.join(data_path, file_dict_r), 'rb') as file:
    Sem_dict_r = cPickle.load(file)

emb_lst1 = np.load(os.path.join(data_path, file_embeddings))

emb_lst = emb_lst1.tolist()

with open(os.path.join(data_path, 'train_15.pickle'), 'rb') as file:
    train_15_l = cPickle.load(file)

wn_dic  = lst_2_dic(WordNet_44_categories)
pos_dic = lst_2_dic(POS_15_categories)
gr_dic  = lst_2_dic(GR_19_categories)



wn_dic['paddd'] = 0
pos_dic['paddd'] = 0
gr_dic['paddd'] = 0

#vocab_dic  = load_wrd_vec_dic_v2(path_raw_data)
#print(vocab_dic['paddd'])
vocab_dic['.'] = [len(vocab_dic), np.zeros(100)]
emb_lst.append(np.zeros(100))

#emb_lst    = load_emd_lst_v2(path_raw_data)
#emb_lst = np.append(emb_lst, np.zeros(shape=(1,100)), axis=0)
#emb_lst.append(np.random.uniform(-0.25, 0.25, 200))

print(len(emb_lst))
print(len(vocab_dic))


files = ['train.txt', 'test.txt']

words = {}
maxSentenceLen = [0,0]


distanceMapping = {'PADDING': 0, 'LowerMin': 1, 'GreaterMax': 2}

# Assigning distances a embedding index

minDistance = -30
maxDistance = 30
for dis in range(minDistance,maxDistance+1):
    #print(dis)
    distanceMapping[dis] = len(distanceMapping)
    #print(dis, distanceMapping[dis])

def getWordIdx(token): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in vocab_dic:
        return vocab_dic[token][0]
    elif token.lower() in vocab_dic:
        return vocab_dic[token.lower()][0]
    #unkn = unkn + 1
    #print(token)
    v1 = np.random.uniform(-0.25, 0.25, 100)
    vocab_dic[token] = [len(vocab_dic), v1]

    #emb_lst = np.append(emb_lst, np.reshape(v1, (1, 100)), axis=0)
    emb_lst.append(v1)
    #print(len(emb_lst))
    #emb_lst = emb_lst1
    return vocab_dic[token][0]


def createMatrices(file, maxSentenceLen=97):
    """Creates matrices for the events and sentence for the given file"""
    
    #labels = []
    #unkn1 = 0

    positionMatrix1 = []    # Distances of the words from e1
    positionMatrix2 = []
    tokenMatrix = []        # Word embeddings (Use Word2vec, glove, dependency embeddings)
    
    
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        pos1 = splits[1]
        pos2 = splits[2]
        sentence = splits[3]
        tokens = sentence.split(" ")
        
       
      
        
        tokenIds = np.zeros(maxSentenceLen)   # 0 is not the padding index actually its 26463
        tokenIds.fill(22295)
        positionValues1 = np.zeros(maxSentenceLen)   # Is 0 the padding here for position vector ??? 
        positionValues2 = np.zeros(maxSentenceLen)
        
        for idx in range(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx])
            
            #if tokenIds[idx] == 26464:
            #    unkn1 += 1
            #    print(tokens[idx])

            distance1 = idx - int(pos1)
            distance2 = idx - int(pos2)
            
            if distance1 in distanceMapping:
                positionValues1[idx] = distanceMapping[distance1]
            elif distance1 <= minDistance:
                positionValues1[idx] = distanceMapping['LowerMin']
            else:
                positionValues1[idx] = distanceMapping['GreaterMax']
                
            if distance2 in distanceMapping:
                positionValues2[idx] = distanceMapping[distance2]
            elif distance2 <= minDistance:
                positionValues2[idx] = distanceMapping['LowerMin']
            else:
                positionValues2[idx] = distanceMapping['GreaterMax']
        #print(tokens)
        #print(tokenIds)
        #print(positionValues1)
        #print(positionValues2)    
        #exit()
        tokenMatrix.append(tokenIds)
        positionMatrix1.append(positionValues1)
        positionMatrix2.append(positionValues2)
        
        #labels.append(labelsMapping[label])
        

    #print(unkn1)
    return np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
        
        
        
 




            

train_set = createMatrices(files[0],  97)
test_set = createMatrices(files[1],  97)

print(len(vocab_dic))
print(len(emb_lst))

print(vocab_dic['.'])


print(train_set[0][0])
print(train_set[1][0])


l1 = ['The', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 'arrayed', 'configuration', 'of', 'antenna', 'elements', '.']

# check : seems fine
'''
for i in l1:
	print(vocab_dic[i.lower()][0])
'''
##############################################3
#  Take care of data augmentation
######################################################3

aug = {} # Int : int 

aug[1] = 1 
aug[2] = 2
aug[3] = 3
aug[4] = 4
aug[5] = 5
aug[6] = 6
aug[7] = 7
aug[8] = 8
aug[9] = 9
aug[10] = 10

with open('SDP_for_8000.txt','r') as f1:
	forw = f1.readlines()
with open('SDP_back_800.txt','r') as f2:
	back = f2.readlines()


for i in xrange(11, len(forw)+1):
	l = forw[i-1].strip()
	for k in xrange(aug[i-1], len(back)):
		l1 = back[k].strip()
		if l == l1:
			aug[i] = k + 1
			break


#print (aug)
print(len(aug))

aug_train = []
aug_pos1 = []
aug_pos2 = []

for i in xrange(1, 5914):
	aug_idx = 799 + aug[i]  # 0 based indexing
	pos1 = train_set[2][aug_idx]
	pos2 = train_set[1][aug_idx]
	tokenIds = train_set[0][aug_idx]
	aug_train.append(tokenIds)
	aug_pos1.append(pos1)
	aug_pos2.append(pos2)

aug_train = np.array(aug_train)
aug_pos1 = np.array(aug_pos1)
aug_pos2 = np.array(aug_pos2)

train_set_words = np.concatenate((train_set[0], aug_train),axis=0)
train_set_pos1 = np.concatenate((train_set[1], aug_pos1),axis=0)
train_set_pos2 = np.concatenate((train_set[2], aug_pos2),axis=0)

train_set_aug = train_set_words, train_set_pos1, train_set_pos2

print(train_set_aug[0].shape)





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
            item = item.lower()
            if item not in vocab_dic:
                word_index = len(vocab_dic)
                v1 = np.random.uniform(-0.25, 0.25, 100)
                vocab_dic[item] = [len(vocab_dic), v1]
                emb_lst.append(v1)
            else:    
                word_index = vocab_dic[item][0]
            train_ex.append(word_index)

        train_ex_rev = []    
        for item in SPT[5]:
            item = item.lower()
            if item not in vocab_dic:
                word_index = len(vocab_dic)
                v1 = np.random.uniform(-0.25, 0.25, 100)
                vocab_dic[item] = [len(vocab_dic), v1]
                emb_lst.append(v1)
            else:    
                word_index = vocab_dic[item][0]
            train_ex_rev.append(word_index)

        if len(train_ex) < truncated_backprop_length:
            start = len(train_ex)
            for i in range(start, truncated_backprop_length):
                train_ex.append(22295)

        if len(train_ex_rev) < truncated_backprop_length:
            start = len(train_ex_rev)
            for i in range(start, truncated_backprop_length):
                train_ex_rev.append(22295)

        batchX_train_forward.append(train_ex)
        batchX_train_reverse.append(train_ex_rev)
    
    
    batchX_test_forward = []
    batchX_test_reverse = []
    for k in xrange(test_b, test_e):
        SPT = SPTs_test[k]
 
        test_ex = []
        for item in SPT[4]:
            item = item.lower()
            if item not in vocab_dic:
                word_index = len(vocab_dic)
                v1 = np.random.uniform(-0.25, 0.25, 100)
                vocab_dic[item] = [len(vocab_dic), v1]
                emb_lst.append(v1)
            else:    
                word_index = vocab_dic[item.lower()][0]
            test_ex.append(word_index)

        test_ex_rev = []    
        for item in SPT[5]:
            item = item.lower()
            if item not in vocab_dic:
                word_index = len(vocab_dic)
                v1 = np.random.uniform(-0.25, 0.25, 100)
                vocab_dic[item] = [len(vocab_dic), v1]
                emb_lst.append(v1)
            else:    
                word_index = vocab_dic[item.lower()][0]
            test_ex_rev.append(word_index)

        if len(test_ex) < truncated_backprop_length:
            start = len(test_ex)
            for i in range(start, truncated_backprop_length):
                test_ex.append(22295)

        if len(test_ex_rev) < truncated_backprop_length:
            start = len(test_ex_rev)
            for i in range(start, truncated_backprop_length):
                test_ex_rev.append(22295)

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

#################################################################################################################################
#
x_forward, x_backward, y_train, x_test_f, x_test_r, y_test = build_SDPs_RNNs(train_b, train_e, valid_b, valid_e, test_b, test_e)
POS_X_train_f, POS_X_train_r, Wnet_X_train_f, Wnet_X_train_r, GR_X_train_f,GR_X_train_r = load_POS_GR_Wnet_train(train_b, train_e, valid_b, valid_e, test_b, test_e)
POS_X_test_f, POS_X_test_r, Wnet_X_test_f, Wnet_X_test_r, GR_X_test_f, GR_X_test_r  =  load_POS_GR_Wnet_test(train_b, train_e, valid_b, valid_e, test_b, test_e)   

x_train_left = np.array(x_forward)
x_train_right = np.array(x_backward)
x_test_left = np.array(x_test_f)
x_test_right = np.array(x_test_r)


print(train_set_aug[0].shape)
print(test_set[0].shape)
print(x_train_left.shape)
print(x_train_right.shape)
print(x_test_left.shape)
print(x_test_right.shape)
print(POS_X_train_f.shape)
print(POS_X_train_r.shape)


Embeddings = np.array(emb_lst)

data = {'Embeddings': Embeddings, 'word_dict': vocab_dic, 'Emb_list': emb_lst, 
        'train_raw': train_set_aug, 'test_raw': test_set,
        'x_train_left': x_train_left, 'x_train_right': x_train_right,
        'x_test_left': x_test_left, 'x_test_right': x_test_right,
        'POS_X_train_f': POS_X_train_f, 'POS_X_train_r':POS_X_train_r, 
        'Wnet_X_train_f': Wnet_X_train_f, 'Wnet_X_train_r':Wnet_X_train_r, 
        'GR_X_train_f': GR_X_train_f, 'GR_X_train_r': GR_X_train_r,
        'POS_X_test_f': POS_X_test_f, 'POS_X_test_r': POS_X_test_r, 
        'Wnet_X_test_f': Wnet_X_test_f, 'Wnet_X_test_r':Wnet_X_test_r, 
        'GR_X_test_f': GR_X_test_f, 'GR_X_test_r': GR_X_test_r,
        'y_train': y_train, 'y_test': y_test
        }

outputFilePath = './sem-raw-SDP_100.pkl.gz'

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()



print(" Raw Data stored in pkl folder")
























#####################################################################
#  Try other Embeddings as well
#####################################################################

       
