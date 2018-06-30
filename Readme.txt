##################################### Program ######################################

This program is about relation classification on SemEval2010-task8.

1. Construct a 4 layer DRNNs using 4channel_4rnn_5pool.py(within package rnn) in the PyDev project Recly0.11.
2. Copy the constructed DRNNs data from Recly0.11 to the project DeepNN011(writen in C++ for efficiency).
3. Train and test DRNNs in DeepNN011.
4. All preprocessed data of SemEval2010-task8 are stored in "Recly0.11/Nets/".

############################### Experimental Results ###############################

Best experimental results of test samples are shown in the "Best test output.txt". 
The final chosen test F1-score is 85.85% selected by validation at 31st epoch, with the highest validation F1-score of 87.17%.
It is worth to note that the best test F1-score among all test values in this file is 86.19%.

