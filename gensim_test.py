import numpy as np
import os
from gensim import corpora, models, similarities
import pickle


path_glove = './ReCly0.11/data/glove.6B/'
glove_file = 'glove.6B.100d.txt'

def loadGloveModel():
    print "Loading Glove Model"
    f = open(os.path.join(path_glove, glove_file),'r')
    model = {}

    for line in f:
        #print len(line)
        splitLine = line.split()
        word = splitLine[0]
        #exit()
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    #exit()
    #print model['e2']
    model['/e1'] = np.random.randn(100)
    model['/e2'] = np.random.randn(100)   
    model['carbohydrase'] = np.random.randn(100)
    model['vasoepididymostomy'] = (-1) * model['vasectomy']
    model['demagnetization']	= (-1) * model['magnetization']	
    model['berserks']  = model['berserkers']
    model['albinistic'] = model['albinism'] + model['adjective']
    model['cyberspaces'] = model['cyberspace'] + model['s']
    model['doylt'] = model['swine']
    model['aeronomist'] = model['aeronomy'] + model['person']
    model['bullfinches'] = model['bullfinch'] + model['es']
    model['abridger'] = model['abridge'] + model['person']
    model['senatorships'] = model['senator'] + model['scholarship'] - model['scholar'] + model['s']
    model['ventilations'] = model['ventilation'] + model['s']
    model['shirtwaist'] = model['shirtwaist'] + model['s']
    model['peakeffect'] = model['effect'] + model['peak'] + model['noun']
    model['hypocalcemic'] = model['hypocalcemia'] + model['anaemic'] - model['anaemia']
    model['gambolling'] = model['dancing']

    model['commissures'] = model['commissure'] + model['girls'] - model['girl']
    model['cosmopolites'] = model['cosmopolite'] + model['girls'] - model['girl']
    model['verbalizations'] = model['verbalization'] + model['girls'] - model['girl']
    model['parings'] = model['paring'] + model['girls'] - model['girl']
    model['pochards'] = model['pochard'] + model['girls'] - model['girl']

    model['gossipers'] = model['gossiper'] + model['girls'] - model['girl']
    model['amphibolites'] = model['amphibolite'] + model['girls'] - model['girl']
    model['recommendations'] = model['recommendation'] + model['girls'] - model['girl']
    #model['osteoarthitis'] = model['arthritis'] + model['osteo'] 
    model['humanistics'] = model['humanistic'] 
    #model['ohmmeters'] = model['ohmmeter'] + model['girls'] - model['girl']
    model['bytecodes'] = model['bytecode'] + model['girls'] - model['girl']
    model['smuck'] = model['penis']
    model['veloutes'] = model['veloute'] + model['girls'] - model['girl']
    model['subluxations'] = model['subluxation'] + model['girls'] - model['girl']
    model['wildfowls'] = model['wildfowl'] + model['girls'] - model['girl']
    #model['disintegrants'] = model['disintegrant'] + model['girls'] - model['girl']
    #model['spoofers'] = model['spoofer'] + model['girls'] - model['girl']
    model['whetstones'] = model['whetstone'] + model['girls'] - model['girl']
    model['architecti'] = model['architect']
    model['tispy'] = model['tipsy']
    #model['bitstrings'] = model['bitstring'] + model['girls'] - model['girl']   binderies 
    model['conies'] = model['coney'] + model['girls'] - model['girl']
    model['dilations'] = model['dilation'] + model['girls'] - model['girl']
    model['opinions'] = model['opinion'] + model['girls'] - model['girl']
    model['shoulders'] = model['shoulder'] + model['girls'] - model['girl']
    model['garbages'] = model['garbage'] + model['girls'] - model['girl']
    model['midranges'] = model['midrange'] + model['girls'] - model['girl']
    model['toolkits'] = model['toolkit'] + model['girls'] - model['girl']
    model['recepies'] = model['recipe'] + model['girls'] - model['girl']
    model['holders'] = model['holder'] + model['girls'] - model['girl']
    model['hippypotamuses'] = model['hippopotamus'] + model['girls'] - model['girl']
    model['physicians'] = model['physician'] + model['girls'] - model['girl']
    model['tumours'] = model['tumour'] + model['girls'] - model['girl']
    model['buildings'] = model['building'] + model['girls'] - model['girl']
    #model['maireeners'] = model['maireener'] + model['girls'] - model['girl'] 
    model['cottes'] = model['cotte'] + model['girls'] - model['girl']
    model['fields'] = model['field'] + model['girls'] - model['girl']
    model['binderies'] = model['bindery'] + model['girls'] - model['girl']
    model['urodiles'] = model['amphibian'] + model['girls'] - model['girl']
    model['entrapments'] = model['entrapment'] + model['girls'] - model['girl']
    model['anurans'] = model['anuran'] + model['girls'] - model['girl']
    model['meinie'] = model['group']
    model['oestrogens'] = model['oestrogen'] + model['girls'] - model['girl']
    model['skegs'] = model['skeg'] + model['girls'] - model['girl'] 
    #model['modelmakers'] = model['modelmaker'] + model['girls'] - model['girl']
    model['furuncles'] = model['boils']
    model['canaille'] = model['collection']  
    model['suckles'] = model['suckle'] + model['goes'] - model['go']
    model['uprises'] = model['uprise'] + model['goes'] - model['go']
    model['profligates'] = model['profligate'] + model['goes'] - model['go']
    model['neutralizes'] = model['neutralize'] + model['goes'] - model['go']
    model['fluther'] = model['group']
    model['clowder'] = model['group']
    model['canaille'] = model['masses']
    model['pleiad'] = model['septet'] 
    model['emitting'] = model['emit'] + model['running'] - model['run']
    model['nectaring'] = model['nectar'] + model['running'] - model['run']
    model['kookiest'] = model['kooky'] + model['best'] - model['good']
    model['affrighted'] = model['frightened']
    #print model['suckler']  
    model['abatoir'] = model['slaughterhouse']
    print "Done.",len(model)," words loaded!"
    with open('dict_glove.pickle', 'wb') as handle:
    	pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #print model
    #return model

loadGloveModel()    
