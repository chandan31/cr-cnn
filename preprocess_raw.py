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

print('Hi')
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

vocab_dic  = load_wrd_vec_dic_v2(path_raw_data)

vocab_dic['paddd'] = [len(vocab_dic), np.zeros(200)]

#vocab_dic['UNKNOWN_TOKEN'] = [len(vocab_dic), np.random.uniform(-0.25, 0.25, 200)]

emb_lst    = load_emd_lst_v2(path_raw_data)
emb_lst.append(np.zeros(200))
#emb_lst.append(np.random.uniform(-0.25, 0.25, 200))



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
    v1 = np.random.uniform(-0.25, 0.25, 200)
    vocab_dic[token] = [len(vocab_dic), v1]
    emb_lst.append(v1)

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
        tokenIds.fill(26463)
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
Embeddings = np.array(emb_lst)
#print('No of unknowns', unk)


print(train_set[0].shape)
print(train_set[1].shape)
print(train_set[2].shape)
print(test_set[0].shape)
print(test_set[1].shape)
print(test_set[2].shape)
print('\n')

l1 = ['The', 'system', 'as', 'described', 'above', 'has', 'its', 'greatest', 'application', 'in', 'an', 'arrayed', 'configuration', 'of', 'antenna', 'elements', '.']

# check : seems fine
'''
for i in l1:
	print(vocab_dic[i][0])
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
data = {'Embeddings': Embeddings, 'word_dict': vocab_dic, 
        'train_set': train_set_aug, 'test_set': test_set}

outputFilePath = './sem-relations_1.pkl.gz'

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()



print("Data stored in pkl folder")

#####################################################################
#  Try other Embeddings as well
#####################################################################

       
