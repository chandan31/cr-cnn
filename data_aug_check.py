import numpy as np
import os
import cPickle


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


print aug
print(len(aug))

#print forw	