# coding: utf-8
from __future__ import print_function

from ROC5 import ROC
from LSTM_CNN import LSTM_CNN_0, LSTM_CNN_1, LSTM_CNN_2, LSTM_CNN_23, LSTM_CNN_3

i = 1

mode = ['FULLtranscript', 'maturemRNA']
type = ["A549","CD8T","HEK293_abacm","HEK293_sysy","HeLa","MOLM13"]

for k in range(6):
    
    # path = "/home/guhua/Documents/BIO/model_data/" + mode[i] + "/embedding/"
    # path = "/home/guhua/Documents/BIO/model_data/" + mode[i] + "/onehot/"
    # path = "/home/guhua/Documents/BIO/model_data/" + mode[i] + "/seq/"


    # train  
    LSTM_CNN_0(data_mode = mode[i], data_type = type[k], n_input=3, training_iters=10001)
    LSTM_CNN_1(data_mode = mode[i], data_type = type[k], n_input=3, training_iters=10001)
    LSTM_CNN_2(data_mode = mode[i], data_type = type[k], n_input=3, training_iters=10001)
    LSTM_CNN_23(data_mode = mode[i], data_type = type[k], n_input=3, training_iters=10001)
    LSTM_CNN_3(data_mode = mode[i], data_type = type[k], n_input=3, training_iters=10001)

    # tmp = path + type[k] + '/'
    # print(tmp)
    # AUC, _, _ = ROC(tmp, iter=100, low=0.10, high=0.50)

