import os 
import numpy as np

l_rate = []
with open('Best test output.txt', 'r') as f:
	content = f.readlines()
	i = 15
	while i < 293:
		l = content[i].split(': ')[1]  #.split('learning rate:')
		l_rate.append(float(l))

		i += 7

print l_rate

			