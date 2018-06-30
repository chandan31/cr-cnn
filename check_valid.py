import numpy as np 
import os

with open('rec_cnv_3_2_valid_preds', 'r') as f:
	preds = f.readlines()

with open('test_target_modified', 'r') as f1:
	true = f1.readlines()

p = []
t = []
for i in preds:
	p.append(i.strip())

for i in true:
	t.append(i.strip())
n_correct = 0
for i in xrange(len(p)):
	if p[i] == t[i]:
		n_correct += 1

print n_correct		
