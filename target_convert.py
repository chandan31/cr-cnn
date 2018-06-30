import os
import numpy as np

conv = {}

 









conv['18'] = 17
conv['7'] = 4	
conv['2'] = 2
conv['19'] = 5
conv['9'] = 18
conv['16'] = 3 
conv['5'] = 12
conv['11'] = 1
conv['3'] = 14
conv['10'] = 6
conv['17'] = 15
conv['1'] = 19
conv['6'] = 10
conv['4'] = 8
conv['15'] = 9
conv['12'] = 13
conv['14'] = 11
conv['13'] = 7
conv['8'] = 16



'''
vals = []
target = []



with open('test_target', 'r') as f:
    content = f.readlines()



for c in content:
	v = c.strip()
	vals.append(conv[v])
	target.append(int(conv[v]))    

with open('test_target_modified', 'w') as f:
	for item in vals:
 		f.write("%s\n" % item)
'''

vals = []
target = []



with open('valid_targets', 'r') as f:
    content = f.readlines()



for c in content:
	v = c.strip()
	vals.append(conv[v])
	target.append(int(conv[v]))    

with open('valid_targets_modified', 'w') as f:
	for item in vals:
 		f.write("%s\n" % item)



'''
preds = []
n_correct = 0
with open('tf_rnn_cpoy6_copy_predictions', 'r') as f:
    content = f.readlines()
    for c in content:
    	c = c.strip()
    	preds.append(int(c))

preds = np.array(preds)
target = np.array(target)

for i in xrange(2717):
	if preds[i] == target[i]:
		n_correct += 1

print n_correct
'''