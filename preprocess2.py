"""
The file preprocesses the files/train.txt and files/test.txt files.

@Chandan
Now I have modified the indexes of category to use torch_eval.py

"""
from __future__ import print_function
import numpy as np
import gzip
import os
import sys
if (sys.version_info > (3, 0)):
    import pickle as pkl
else: #Python 2.7 imports
    import cPickle as pkl

outputFilePath = 'pkl/sem-relations_1.pkl.gz'


#We download English word embeddings from here https://www.cs.york.ac.uk/nlp/extvec/
embeddingsPath = 'embeddings/wiki_extvec.gz'


folder = 'files/'
files = [folder+'train.txt', folder+'test.txt']

#Mapping of the labels to integers
labelsMapping = {'Other':0, 
                 'Message-Topic(e1,e2)':8, 'Message-Topic(e2,e1)':17, 
                 'Product-Producer(e1,e2)':9, 'Product-Producer(e2,e1)':18, 
                 'Instrument-Agency(e1,e2)':6, 'Instrument-Agency(e2,e1)':15, 
                 'Entity-Destination(e1,e2)':4, 'Entity-Destination(e2,e1)':13,
                 'Cause-Effect(e1,e2)':1, 'Cause-Effect(e2,e1)':10,
                 'Component-Whole(e1,e2)':2, 'Component-Whole(e2,e1)':11,  
                 'Entity-Origin(e1,e2)':5, 'Entity-Origin(e2,e1)':14,
                 'Member-Collection(e1,e2)':7, 'Member-Collection(e2,e1)':16,
                 'Content-Container(e1,e2)':3, 'Content-Container(e2,e1)':12}




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


def createMatrices(file, word2Idx, maxSentenceLen=100):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
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
        
       
      
        
        tokenIds = np.zeros(maxSentenceLen)
        positionValues1 = np.zeros(maxSentenceLen)
        positionValues2 = np.zeros(maxSentenceLen)
        
        for idx in range(0, min(maxSentenceLen, len(tokens))):
            tokenIds[idx] = getWordIdx(tokens[idx], word2Idx)
            
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
        
        labels.append(labelsMapping[label])
        

    
    return np.array(labels, dtype='int32'), np.array(tokenMatrix, dtype='int32'), np.array(positionMatrix1, dtype='int32'), np.array(positionMatrix2, dtype='int32'),
        
        
        
 
def getWordIdx(token, word2Idx): 
    """Returns from the word2Idex table the word index for a given token"""       
    if token in word2Idx:
        return word2Idx[token]
    elif token.lower() in word2Idx:
        return word2Idx[token.lower()]
    
    return word2Idx["UNKNOWN_TOKEN"]


##########################################################################################
#   Finding max sentence length and preparing the dictionary 'word' of all the words 
###########################################################################################


for fileIdx in range(len(files)):
    file = files[fileIdx]
    #print(file)
    #print(fileIdx)
    for line in open(file):
        splits = line.strip().split('\t')
        
        label = splits[0]
        
        
        sentence = splits[3]        
        tokens = sentence.split(" ")
        maxSentenceLen[fileIdx] = max(maxSentenceLen[fileIdx], len(tokens))
        #if len(tokens) == 97:
        #    print (tokens)
        for token in tokens:
            words[token.lower()] = True
            

print("Max Sentence Lengths: ", maxSentenceLen)

# :: Read in word embeddings ::
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []


#####################################################################
#  Try other Embeddings as well
#####################################################################

# :: Downloads the embeddings from the York webserver ::
if not os.path.isfile(embeddingsPath):
    basename = os.path.basename(embeddingsPath)
    if basename == 'wiki_extvec.gz':
           print("Start downloading word embeddings for English using wget ...")
           #os.system("wget https://www.cs.york.ac.uk/nlp/extvec/"+basename+" -P embeddings/")
           os.system("wget https://public.ukp.informatik.tu-darmstadt.de/reimers/2017_english_embeddings/"+basename+" -P embeddings/")
    else:
        print(embeddingsPath, "does not exist. Please provide pre-trained embeddings")
        exit()



        
# :: Load the pre-trained embeddings file ::
fEmbeddings = gzip.open(embeddingsPath, "r") if embeddingsPath.endswith('.gz') else open(embeddingsPath, encoding="utf8")
i = 0   
print("Load pre-trained embeddings file")
for line in fEmbeddings:
    split = line.decode('utf-8').strip().split(" ")
    word = split[0]
    #print(split[0], split)
    #exit()
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if word.lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[word] = len(word2Idx)
        i += 1
       
wordEmbeddings = np.array(wordEmbeddings)

print("Embeddings shape: ", wordEmbeddings.shape)
print("Len words: ", len(words))
########
#   Too many unknown words 
#############################################################

# :: Create token matrix ::
train_set = createMatrices(files[0], word2Idx, max(maxSentenceLen))
test_set = createMatrices(files[1], word2Idx, max(maxSentenceLen))



data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 
        'train_set': train_set, 'test_set': test_set}

f = gzip.open(outputFilePath, 'wb')
pkl.dump(data, f)
f.close()



print("Data stored in pkl folder")

################################################################        
#   Problems and editions:
#       1. Use glove etc. less no of unknown words 
#       2. See what are unknown words
#       3. 
#         
################################################################