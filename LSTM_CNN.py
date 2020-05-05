# coding: utf-8
from __future__ import print_function

import os
import csv
import random
from ROC import ROC
import tensorflow as tf
from numpy import argmax
from itertools import chain
from datetime import datetime

record_step = 100
coding_method = ['/onehot','/embedding', '/seq']
method = coding_method[2]

file_data = '/home/guhua/Documents/BIO'
path_embed = '/home/guhua/Documents/BIO/Encoding_data/seq_data/'
# path_embed = '/home/guhua/Documents/BIO/Encoding_data/onehot_data/'
# path_embed = '/home/guhua/Documents/BIO/Encoding_data/embedding_data/'


# read embedding data
with open(path_embed+'rna_embedding.csv', encoding="utf-8") as f:
    reader = csv.reader(f)
    final_embeddings = list(reader)
    print(len(final_embeddings[1]))

with open(path_embed+'rna_dictionary.csv', encoding = "utf-8") as f:
    reader = csv.DictReader(f)
    for i in reader:
        dictionary = dict(i)

# generate batch data
class mRNA_sequence(object):
    def __init__(self, type_data = 'train', data_path_tmp=''):
        # data 
        # test
        self.num_neg = 0
        self.num_pos = 0
        self.pos_data = []
        self.pos_labels = []
        self.neg_data = []
        self.neg_labels = []

        path1 = ['train','test']
        path2 = ['Neg','Pos']
        path_all = []

        for i in path1:
            for j in path2:
                path_data_1 = i + j + '.csv'
                path_all.append(path_data_1)

        # build the dataset
        for i in path_all:
            if type_data in i:
                with open( data_path_tmp + type_data + '/' + i, encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                    if 'Pos' in i:
                        self.pos_data = reader
                        self.pos_labels = [[1.,0.]]*len(reader)
                    if 'Neg' in i:
                        self.neg_data = reader
                        self.neg_labels = [[0.,1.]]*len(reader)
               
        self.num_neg = len(self.neg_labels)
        self.num_pos = len(self.pos_labels)

        self.batch_id_i = 0
        self.batch_id_j = 0
        
    # rate_pos : pos / ALL
    def data_next(self, batch_size, rate_pos = 0.5):
        tmp_batch_data = []
        batch_labels = []
        batch_data = []

        for _ in range(batch_size):
            # Add a positive sample or a negative sample at random
            if random.random() < rate_pos:
                tmp_batch_data.append(self.pos_data[self.batch_id_i])
                batch_labels.append(self.pos_labels[self.batch_id_i])
                self.batch_id_i+=1
                if self.batch_id_i==len(self.pos_data):
                    self.batch_id_i = 0         
            else:
                tmp_batch_data.append(self.neg_data[self.batch_id_j])
                batch_labels.append(self.neg_labels[self.batch_id_j])
                self.batch_id_j+=1
                if self.batch_id_j==len(self.neg_data):
                    self.batch_id_j = 0
            
        # convert to the embedding data
        for i in tmp_batch_data:
            temp = []
            for j in i:
                temp.append(final_embeddings[int(dictionary[j])])
            batch_data.append(temp)

        return batch_data, batch_labels

## 0+0
def LSTM_CNN_0(data_mode = 'FULLtranscript', data_type = 'A549', n_input = 4, len_seq = 32, learning_rate=1e-3, training_iters = 10001, batch_size=128, display_step=100, iter_size=100, Run_type='train', test_path = ''):

    ## STEP 1 load the data and build the dataset

    data_path = file_data + "/DNA_data/"
    data_path = data_path  + data_mode + '/' + data_type + '/'

    path_model = file_data +"/model_data/" + data_mode
    path_model = path_model + method +'/'  + data_type + "/0AUROC/"
   
    print('Data_path:',  data_path)
    print('Model_path:',  path_model)
    print('Embedding_path:',  path_embed)

    trainset = mRNA_sequence(type_data='train', data_path_tmp=data_path)
    testset = mRNA_sequence(type_data='test', data_path_tmp=data_path)

    ## STEP 2 build the LSTM network

    # LSTM network parameters
    seq_max_len = 41     # the length of seq
    # n_input = 4          # embedding dimension
    n_hidden = len_seq   # the dimension of the LSTM hidden layers
    n_classes = 2        # Number of categories
    n_layers = 1         # lstm layers

    graph1 = tf.Graph()

    with graph1.as_default():
        # x is input，y is output
        # None is batch_size
        with tf.name_scope('LSTMCNN0_inputs'):
            with tf.variable_scope('LSTMCNN0_in'):
                x = tf.placeholder("float", [None, seq_max_len, n_input], name = 'x_input')
                y = tf.placeholder("float", [None, n_classes], name = 'y_input')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
                batch_size_ = tf.placeholder(tf.int32, [], name='batch_size_input')

        weights = {
            'in': tf.Variable(tf.random_normal([n_input, n_hidden]),name='w_in'),
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='w_out')
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b_in'),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_out')
        }

        def dynamicRNN(x, weights, biases):
            # the shape of x： (batch_size, max_seq_len, n_input)
            X = tf.reshape(x, [-1, n_input]) 
            X_in = tf.matmul(X, weights['in']) + biases['in']
            X_in = tf.reshape(X_in, [-1, seq_max_len, n_hidden])
            # 定义一个lstm_cell，隐层的大小为n_hidden
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            
            lstm_cells = []
            for _ in range(n_layers):
                lstm_cells.append(lstm_cell())
            with tf.name_scope('LSTMCNN0_lstm_cells_layers'):
                mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple = True)
            
            # initiate state
            state = mlstm_cell.zero_state(batch_size_,dtype=tf.float32)
            
            # 使用tf.nn.dynamic_rnn展开时间维度
            # outputs的形状为(batch_size, max_seq_len, n_hidden)
            with tf.variable_scope('LSTMCNN0_out'):
                outputs0, _ = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=state, dtype=tf.float32, time_major=False)

            # return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']
            return outputs0

        ## STEP 3 build the CNN network

        # initial weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            return tf.Variable(initial)

        # initial bias
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # convolution layer
        def conv1d(x,f,k):
            return tf.layers.conv1d(x, f, k, padding='VALID')

        # lstm
        with tf.name_scope('LSTMCNN0_lstm_layer'):
            lstm_out = dynamicRNN(x, weights, biases)

        # Conv1
        with tf.name_scope('LSTMCNN0_Conv_1'):
            h_conv1 = conv1d(lstm_out, 64, 4)
            dim = h_conv1.get_shape()[1].value*h_conv1.get_shape()[2].value

        # Fc_1
        with tf.name_scope('LSTMCNN0_Fc_1'):
            with tf.name_scope('resh_1'):
                reshape = tf.reshape(h_conv1,[batch_size_, -1])
            with tf.name_scope('w_fc1'):
                W_fc1 = weight_variable([dim, 256])
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([256])
            with tf.name_scope('relu'):
                h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
            #dropout
            with tf.name_scope('dropout'):
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Softmax
        with tf.name_scope('LSTMCNN0_softmax'):
            with tf.name_scope('w_softmax'):
                W_fc3 = weight_variable([256,n_classes])
            with tf.name_scope('b_softmax'):
                b_fc3 = bias_variable([n_classes])
            with tf.name_scope('prediction_softmax'):
                pred = (tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

        # loss
        with tf.name_scope('LSTMCNN0_loss'):
            cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))
            tf.summary.scalar('loss', cost)

        with tf.name_scope('LSTMCNN0_train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # acc
        with tf.name_scope('LSTMCNN0_accuracy'):
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('acc', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    ## STEP 4 train the network

    if Run_type == 'train':
        with tf.Session(graph = graph1) as sess:
            sess.run(init)
            train_writer = tf.summary.FileWriter(path_model+"logs/train",sess.graph)
            test_writer = tf.summary.FileWriter(path_model+"logs/test",sess.graph)
            step = 1
            while step < training_iters:
                batch_x, batch_y = trainset.data_next(batch_size)
                test_x, test_y = testset.data_next(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5, batch_size_: batch_size})

                if step % record_step == 0:
                    summary_train = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size_:batch_size})
                    summary_test  = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size_:batch_size})
                    train_writer.add_summary(summary_train, step)
                    test_writer.add_summary(summary_test, step)

                if step % display_step == 0:
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0, batch_size_:batch_size})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    print("%s: Step [%d]  Loss : %.6f, training accuracy :  %.5f, test accuracy :  %.5f" % (datetime.now(), step, loss, acc_train, acc_test))
                    
                step += 1
            print("Optimization Finished!")

            # Save model weights to disk
            saver.save(sess, path_model+"data")


        ## ROC CURVE AND AUROC
        path_pred  = path_model + 'predLSTMCNN0.csv'
        path_label = path_model + 'labelLSTMCNN0.csv'

        with tf.Session(graph = graph1) as sess:
            sess.run(init)
            saver.restore(sess, path_model + "data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            pred_neg = []
            for _ in range(10):
                test_data, test_label = testset.data_next(num_pos, rate_pos=0)
                pred_ = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
                pred_neg.extend(pred_)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            # record the real and the pred
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)

        # Draw the ROC curve and get the AUC
        auroc = ROC(path_pred, path_label, iter=iter_size)
        print(auroc)

    # test the network
    elif Run_type == 'test':
        
        testset = mRNA_sequence(type_data='test', data_path_tmp="/")

        path_pred  = '/predLSTMCNN0.csv'
        path_label = '/labelLSTMCNN0.csv'

        with tf.Session(graph = graph1) as sess:
            sess.run(init)
            saver.restore(sess, test_path + "/0AUROC/model_data/data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)


            test_data, test_label = testset.data_next(num_neg, rate_pos=0)
            pred_neg = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_neg})
            pred_neg = list(pred_neg)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)

        auroc = ROC(path_pred, path_label, iter=iter_size)


## 0+1 all
def LSTM_CNN_1(data_mode = 'FULLtranscript', data_type = 'A549', n_input = 4, len_seq = 32, learning_rate=1e-3, training_iters = 10001, batch_size=128, display_step=100,iter_size=100, Run_type='train', test_path = ''):

    ## STEP 1 load the data and build the dataset

    # file_data = os.getcwd()

    data_path = file_data + "/DNA_data/"
    data_path = data_path  + data_mode + '/' + data_type + '/'

    path_model = file_data +"/model_data/" + data_mode
    path_model = path_model + method + '/'  + data_type + "/1AUROC/"

    print('Data_path:',  data_path)
    print('Model_path:',  path_model)
    print('Embedding_path:',  path_embed)

    trainset = mRNA_sequence(type_data='train', data_path_tmp=data_path)
    testset = mRNA_sequence(type_data='test', data_path_tmp=data_path)

    ## STEP 2 build the lstm network

    # LSTM network parameters
    seq_max_len = 41     # the length of seq
    # n_input = 4          # embedding dimension
    n_hidden = len_seq   # the dimension of the LSTM hidden layers
    n_classes = 2        # Number of categories
    n_layers = 1         # lstm layers
    
    graph2 = tf.Graph()

    with graph2.as_default():
        # x is input，y is output
        # None is batch_size
        with tf.name_scope('LSTMCNN1_inputs'):
            with tf.variable_scope('LSTMCNN1_in'):
                x = tf.placeholder("float", [None, seq_max_len, n_input], name = 'x_input')
                y = tf.placeholder("float", [None, n_classes], name = 'y_input')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
                batch_size_ = tf.placeholder(tf.int32, [], name='batch_size_input')

        weights = {
            'in': tf.Variable(tf.random_normal([n_input, n_hidden]),name='w_in'),
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='w_out')
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b_in'),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_out')
        }

        def dynamicRNN(x, weights, biases):
            # the shape of x： (batch_size, max_seq_len, n_input)
            X = tf.reshape(x, [-1, n_input])
            X_in = tf.matmul(X, weights['in']) + biases['in']
            X_in = tf.reshape(X_in, [-1, seq_max_len, n_hidden])
            # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            
            lstm_cells = []
            for _ in range(n_layers):
                lstm_cells.append(lstm_cell())
            with tf.name_scope('LSTMCNN1_lstm_cells_layers'):
                mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple = True)
            
            # initiate state
            state = mlstm_cell.zero_state(batch_size_,dtype=tf.float32)
            
            # 使用tf.nn.dynamic_rnn展开时间维度
            # outputs的形状为(batch_size, max_seq_len, n_hidden)
            with tf.variable_scope('LSTMCNN1_out'):
                outputs1, _ = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=state, dtype=tf.float32, time_major=False)

            # 只需要最后一个时间维度的输出
            # return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']
            return outputs1

        ## STEP 3 build the CNN network

        # initial weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            return tf.Variable(initial)

        # initial bias
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # convolution layer
        def conv1d(x,f,k):
            return tf.layers.conv1d(x, f, k, padding='VALID')

        # pred is logits
        with tf.name_scope('LSTMCNN1_lstm_layer'):
            lstm_out = dynamicRNN(x, weights, biases)

        # Conv1
        with tf.name_scope('LSTMCNN1_Conv_1'):
            h_conv1 = conv1d(lstm_out, 64, 4)
            dim = h_conv1.get_shape()[1].value*h_conv1.get_shape()[2].value

        # Fc_1
        with tf.name_scope('LSTMCNN1_Fc_1'):
            with tf.name_scope('resh_1'):
                reshape = tf.reshape(h_conv1,[batch_size_, -1])
            with tf.name_scope('w_fc1'):
                W_fc1 = weight_variable([dim, 256])
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([256])
            with tf.name_scope('relu'):
                h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
            #dropout
            with tf.name_scope('dropout'):
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Softmax
        with tf.name_scope('LSTMCNN1_softmax'):
            with tf.name_scope('w_softmax'):
                W_fc3 = weight_variable([256,n_classes])
            with tf.name_scope('b_softmax'):
                b_fc3 = bias_variable([n_classes])
            with tf.name_scope('prediction_softmax'):
                pred1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

        pred2 = tf.matmul(lstm_out[:,-1,:], weights['out']) + biases['out']
        pred = pred1+pred2

        # loss
        with tf.name_scope('LSTMCNN1_loss'):
            cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred1, labels=y))
            cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred2, labels=y))
            cost = (cost1+cost2)/2
            tf.summary.scalar('loss', cost)

        with tf.name_scope('LSTMCNN1_train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # acc
        with tf.name_scope('LSTMCNN1_accuracy'):
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('acc', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    ## STEP 4 train the network

    if Run_type == 'train':
        with tf.Session(graph = graph2) as sess:
            sess.run(init)
            train_writer = tf.summary.FileWriter(path_model+"logs/train",sess.graph)
            test_writer = tf.summary.FileWriter(path_model+"logs/test",sess.graph)
            step = 1
            while step < training_iters:
                batch_x, batch_y = trainset.data_next(batch_size)
                test_x, test_y = testset.data_next(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5, batch_size_: batch_size})

                if step % record_step == 0:
                    summary_train = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size_:batch_size})
                    summary_test  = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size_:batch_size})
                    train_writer.add_summary(summary_train, step)
                    test_writer.add_summary(summary_test, step)

                if step % display_step == 0:
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0, batch_size_:batch_size})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    print("%s: Step [%d]  Loss : %.6f, training accuracy :  %.5f, test accuracy :  %.5f" % (datetime.now(), step, loss, acc_train, acc_test))
                    
                step += 1
            print("Optimization Finished!")

            # Save model weights to disk
            saver.save(sess, path_model+"data")

        ## ROC CURVE AND AUROC
        path_pred  = path_model + 'predLSTMCNN1.csv'
        path_label = path_model + 'labelLSTMCNN1.csv'

        with tf.Session(graph = graph2) as sess:
            sess.run(init)
            saver.restore(sess, path_model + "data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            pred_neg = []
            for i in range(10):
                test_data, test_label = testset.data_next(num_pos, rate_pos=0)
                pred_ = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
                pred_neg.extend(pred_)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            # record the real and the pred
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)


        # Draw the ROC curve and get the AUC
        auroc = ROC(path_pred, path_label, iter=iter_size)
        print(auroc)

    elif Run_type == 'test':
        
        testset = mRNA_sequence(type_data='test', data_path_tmp="/")

        path_pred  = '/predLSTMCNN1.csv'
        path_label = '/labelLSTMCNN1.csv'

        with tf.Session(graph = graph2) as sess:
            sess.run(init)
            saver.restore(sess, test_path + "/1AUROC/model_data/data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)


            test_data, test_label = testset.data_next(num_neg, rate_pos=0)
            pred_neg = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_neg})
            pred_neg = list(pred_neg)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)

        auroc = ROC(path_pred, path_label, iter=iter_size)


## 1+1 all+middle
def LSTM_CNN_2(data_mode = 'FULLtranscript', data_type = 'A549', n_input = 4, len_seq = 32, learning_rate=1e-3, training_iters = 10001, batch_size=128, display_step=100,iter_size=100, Run_type='train', test_path = ''):

    ## STEP 1 load the data and build the dataset

    # file_data = os.getcwd()

    data_path = file_data + "/DNA_data/"
    data_path = data_path  + data_mode + '/' + data_type + '/'

    path_model = file_data +"/model_data/" + data_mode
    path_model = path_model + method + '/'  + data_type + "/2AUROC/"
    
    print('Data_path:',  data_path)
    print('Model_path:',  path_model)
    print('Embedding_path:',  path_embed)
    
    trainset = mRNA_sequence(type_data='train', data_path_tmp=data_path)
    testset = mRNA_sequence(type_data='test', data_path_tmp=data_path)

    ## STEP 2 build the LSTM network

    # LSTM network parameters
    seq_max_len = 41     # the length of seq
    # n_input = 4          # embedding dimension
    n_hidden = len_seq   # the dimension of the LSTM hidden layers
    n_classes = 2        # Number of categories
    n_layers = 1         # lstm layers

    graph3 = tf.Graph()

    with graph3.as_default():
        # x is input，y is output
        # None is batch_size
        with tf.name_scope('LSTMCNN2_inputs'):
            with tf.variable_scope('LSTMCNN2_in'):
                x = tf.placeholder("float", [None, seq_max_len, n_input], name = 'x_input')
                y = tf.placeholder("float", [None, n_classes], name = 'y_input')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
                batch_size_ = tf.placeholder(tf.int32, [], name='batch_size_input')

        weights = {
            'in': tf.Variable(tf.random_normal([n_input, n_hidden]),name='w_in'),
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='w_out')
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b_in'),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_out')
        }

        def dynamicRNN(x, weights, biases):
            # the shape of x： (batch_size, max_seq_len, n_input)
            X = tf.reshape(x, [-1, n_input])
            X_in = tf.matmul(X, weights['in']) + biases['in']
            X_in = tf.reshape(X_in, [-1, seq_max_len, n_hidden])
            # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            
            lstm_cells = []
            for _ in range(n_layers):
                lstm_cells.append(lstm_cell())
            with tf.name_scope('LSTMCNN2_lstm_cells_layers'):
                mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple = True)
            
            # initiate state
            state = mlstm_cell.zero_state(batch_size_,dtype=tf.float32)
            
            # 使用tf.nn.dynamic_rnn展开时间维度
            # outputs的形状为(batch_size, max_seq_len, n_hidden)
            with tf.variable_scope('LSTMCNN2_outs'):
                outputs2, _ = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=state, dtype=tf.float32, time_major=False)

            # 只需要最后一个时间维度的输出
            # return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']
            return outputs2

        ## STEP 3 build the CNN network

        # initial weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            return tf.Variable(initial)

        # initial bias
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # convolution layer
        def conv1d(x,f,k):
            return tf.layers.conv1d(x, f, k, padding='VALID')

        # pred is logits
        with tf.name_scope('LSTMCNN2_lstm_layer'):
            lstm_out = dynamicRNN(x, weights, biases)

        # Conv1
        with tf.name_scope('LSTMCNN2_Conv_1'):
            h_conv1 = conv1d(lstm_out, 64, 4)
            dim = h_conv1.get_shape()[1].value*h_conv1.get_shape()[2].value

        # Fc_1
        with tf.name_scope('LSTMCNN2_Fc_1'):
            with tf.name_scope('resh_1'):
                reshape = tf.reshape(h_conv1,[batch_size_, -1])
            with tf.name_scope('w_fc1'):
                W_fc1 = weight_variable([dim, 256])
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([256])
            with tf.name_scope('relu'):
                h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
            #dropout
            with tf.name_scope('dropout'):
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Softmax
        with tf.name_scope('LSTMCNN2_softmax'):
            with tf.name_scope('w_softmax'):
                W_fc3 = weight_variable([256,n_classes])
            with tf.name_scope('b_softmax'):
                b_fc3 = bias_variable([n_classes])
            with tf.name_scope('prediction_softmax'):
                pred1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

        pred2 = tf.matmul(lstm_out[:,-1,:], weights['out']) + biases['out']
        pred3 = tf.matmul(lstm_out[:,21,:], weights['out']) + biases['out']
        pred = pred1+pred2+pred3

        # loss
        with tf.name_scope('LSTMCNN2_loss'):
            # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.argmax(y,1)))
            cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred1, labels=y))
            cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred2, labels=y))
            cost3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred3, labels=y))
            cost4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

            cost = ((cost1+cost2+cost3)/3 + cost4)/2
            tf.summary.scalar('loss', cost)

        with tf.name_scope('LSTMCNN2_train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # acc
        with tf.name_scope('LSTMCNN2_accuracy'):
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('acc', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    ## STEP 4 train the network

    # 训练
    if Run_type == 'train':
        with tf.Session(graph = graph3) as sess:
            sess.run(init)
            train_writer = tf.summary.FileWriter(path_model+"logs/train",sess.graph)
            test_writer = tf.summary.FileWriter(path_model+"logs/test",sess.graph)
            step = 1
            while step < training_iters:
                batch_x, batch_y = trainset.data_next(batch_size)
                test_x, test_y = testset.data_next(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5, batch_size_: batch_size})

                if step % record_step == 0:
                    summary_train = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size_:batch_size})
                    summary_test  = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size_:batch_size})
                    train_writer.add_summary(summary_train, step)
                    test_writer.add_summary(summary_test, step)

                # 展示 100步一次
                if step % display_step == 0:
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0, batch_size_:batch_size})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    print("%s: Step [%d]  Loss : %.6f, training accuracy :  %.5f, test accuracy :  %.5f" % (datetime.now(), step, loss, acc_train, acc_test))
                    
                step += 1
            print("Optimization Finished!")

            # Save model weights to disk
            saver.save(sess, path_model+"data")

        ## ROC CURVE AND AUROC
        path_pred  = path_model + 'predLSTMCNN2.csv'
        path_label = path_model + 'labelLSTMCNN2.csv'

        # 运行得到预测值和真实值
        with tf.Session(graph = graph3) as sess:
            sess.run(init)
            saver.restore(sess, path_model + "data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            pred_neg = []
            for i in range(10):
                test_data, test_label = testset.data_next(num_pos, rate_pos=0)
                pred_ = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
                pred_neg.extend(pred_)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            # record the real and the pred
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)


        # Draw the ROC curve and get the AUC
        auroc = ROC(path_pred, path_label, iter=iter_size)
        print(auroc)

    elif Run_type == "test":   
        
        testset = mRNA_sequence(type_data='test', data_path_tmp="/")

        path_pred  = '/predLSTMCNN2.csv'
        path_label = '/labelLSTMCNN2.csv'

        with tf.Session(graph = graph3) as sess:
            sess.run(init)
            saver.restore(sess,  test_path + "/2AUROC/model_data/data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            test_data, test_label = testset.data_next(num_neg, rate_pos=0)
            pred_neg = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_neg})
            pred_neg = list(pred_neg)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)

        auroc = ROC(path_pred, path_label, iter=iter_size)


## -1+1 all-middle
def LSTM_CNN_23(data_mode = 'FULLtranscript', data_type = 'A549', n_input = 4, len_seq = 32, learning_rate=1e-3, training_iters = 10001, batch_size=128, display_step=100,iter_size=100, Run_type='train', test_path = ''):

    ## STEP 1 load the data and build the dataset

    # file_data = os.getcwd()

    data_path = file_data + "/DNA_data/"
    data_path = data_path  + data_mode + '/' + data_type + '/'

    path_model = file_data +"/model_data/" + data_mode
    path_model = path_model + method + '/'  + data_type + "/23AUROC/"
    
    print('Data_path:',  data_path)
    print('Model_path:',  path_model)
    print('Embedding_path:',  path_embed)
    
    trainset = mRNA_sequence(type_data='train', data_path_tmp=data_path)
    testset = mRNA_sequence(type_data='test', data_path_tmp=data_path)

    ## STEP 2 build the LSTM network

    # LSTM network parameters
    seq_max_len = 41     # the length of seq
    # n_input = 4          # embedding dimension
    n_hidden = len_seq   # the dimension of the LSTM hidden layers
    n_classes = 2        # Number of categories
    n_layers = 1         # lstm layers

    graph4 = tf.Graph()

    with graph4.as_default():
        # x is input，y is output
        # None is batch_size
        with tf.name_scope('LSTMCNN23_inputs'):
            with tf.variable_scope('LSTMCNN23_in'):
                x = tf.placeholder("float", [None, seq_max_len, n_input], name = 'x_input')
                y = tf.placeholder("float", [None, n_classes], name = 'y_input')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
                batch_size_ = tf.placeholder(tf.int32, [], name='batch_size_input')

        weights = {
            'in': tf.Variable(tf.random_normal([n_input, n_hidden]),name='w_in'), # (30, 64)
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='w_out') # (64, 2)
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b_in'),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_out')
        }

        def dynamicRNN(x, weights, biases):
            # the shape of x： (batch_size, max_seq_len, n_input)
            X = tf.reshape(x, [-1, n_input])
            X_in = tf.matmul(X, weights['in']) + biases['in']
            X_in = tf.reshape(X_in, [-1, seq_max_len, n_hidden])
            # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            
            lstm_cells = []
            for _ in range(n_layers):
                lstm_cells.append(lstm_cell())
            with tf.name_scope('LSTMCNN23_lstm_cells_layers'):
                mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple = True)
            
            # initiate state
            state = mlstm_cell.zero_state(batch_size_,dtype=tf.float32)
            
            # 使用tf.nn.dynamic_rnn展开时间维度
            # outputs的形状为(batch_size, max_seq_len, n_hidden)
            with tf.variable_scope('LSTMCNN23_out'):
                outputs23, _ = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=state, dtype=tf.float32, time_major=False)

            # 只需要最后一个时间维度的输出
            # return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']
            return outputs23

        ## STEP 3 build the CNN network

        # initial weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            return tf.Variable(initial)

        # initial bias
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # convolution layer
        def conv1d(x,f,k):
            return tf.layers.conv1d(x, f, k, padding='VALID')

        # pred is logits
        with tf.name_scope('LSTMCN23_lstm_layer'):
            lstm_out = dynamicRNN(x, weights, biases)

        # Conv1
        with tf.name_scope('LSTMCNN23_Conv_1'):
            h_conv1 = conv1d(lstm_out, 64, 4)
            dim = h_conv1.get_shape()[1].value*h_conv1.get_shape()[2].value

        # Fc_1
        with tf.name_scope('LSTMCNN23_Fc_1'):
            with tf.name_scope('resh_1'):
                reshape = tf.reshape(h_conv1,[batch_size_, -1])
            with tf.name_scope('w_fc1'):
                W_fc1 = weight_variable([dim, 256])
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([256])
            with tf.name_scope('relu'):
                h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
            #dropout
            with tf.name_scope('dropout'):
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Softmax
        with tf.name_scope('LSTMCNN23_softmax'):
            with tf.name_scope('w_softmax'):
                W_fc3 = weight_variable([256,n_classes])
            with tf.name_scope('b_softmax'):
                b_fc3 = bias_variable([n_classes])
            with tf.name_scope('prediction_softmax'):
                pred1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

        pred2 = tf.matmul(lstm_out[:,-1,:], weights['out']) + biases['out']
        pred3 = tf.matmul(lstm_out[:,21,:], weights['out']) + biases['out']
        pred = pred1+pred2-pred3

        # loss
        with tf.name_scope('LSTMCNN23_loss'):
            # cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.argmax(y,1)))
            cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred1, labels=y))
            cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred2, labels=y))
            cost3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred3, labels=y))
            cost4 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

            cost = ((cost1+cost2+cost3)/3 + cost4)/2
            
            tf.summary.scalar('loss', cost)

        with tf.name_scope('LSTMCNN23_train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # acc
        with tf.name_scope('LSTMCNN23_accuracy'):
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('acc', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    ## STEP 4 train the network

    if Run_type == 'train':
        # 训练
        with tf.Session(graph = graph4) as sess:
            sess.run(init)
            train_writer = tf.summary.FileWriter(path_model+"logs/train",sess.graph)
            test_writer = tf.summary.FileWriter(path_model+"logs/test",sess.graph)
            step = 1
            while step < training_iters:
                batch_x, batch_y = trainset.data_next(batch_size)
                test_x, test_y = testset.data_next(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5, batch_size_: batch_size})

                if step % record_step == 0:
                    summary_train = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size_:batch_size})
                    summary_test  = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size_:batch_size})
                    train_writer.add_summary(summary_train, step)
                    test_writer.add_summary(summary_test, step)

                if step % display_step == 0:
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0, batch_size_:batch_size})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    print("%s: Step [%d]  Loss : %.6f, training accuracy :  %.5f, test accuracy :  %.5f" % (datetime.now(), step, loss, acc_train, acc_test))
                    
                step += 1
            print("Optimization Finished!")

            # Save model weights to disk
            saver.save(sess, path_model+"/data")

        ## ROC CURVE AND AUROC
        path_pred  = path_model + 'predLSTMCNN23.csv'
        path_label = path_model + 'labelLSTMCNN23.csv'

        with tf.Session(graph = graph4) as sess:
            sess.run(init)
            saver.restore(sess, path_model + "data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            pred_neg = []
            for i in range(10):
                test_data, test_label = testset.data_next(num_pos, rate_pos=0)
                pred_ = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
                pred_neg.extend(pred_)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            # record the real and the pred
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)


        # Draw the ROC curve and get the AUC
        auroc = ROC(path_pred, path_label, iter=iter_size)
        print(auroc)
    
    elif Run_type == 'test':

        testset = mRNA_sequence(type_data='test', data_path_tmp="")

        path_pred  = '/predLSTMCNN23.csv'
        path_label = '/labelLSTMCNN23.csv'
 
        # 运行得到预测值和真实值
        with tf.Session(graph = graph4) as sess:
            sess.run(init)
            saver.restore(sess, test_path + "/23AUROC/model_data/data")

            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            test_data, test_label = testset.data_next(num_neg, rate_pos=0)
            pred_neg = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_neg})
            pred_neg = list(pred_neg)
            
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)

        auroc = ROC(path_pred, path_label, iter=iter_size)


## 1+0 middle
def LSTM_CNN_3(data_mode = 'FULLtranscript', data_type = 'A549', n_input = 4, len_seq = 32, learning_rate=1e-3, training_iters = 10001, batch_size=128, display_step=100,iter_size=100, Run_type='train', test_path = ''):

    ## STEP 1 load the data and build the dataset

    # file_data = os.getcwd()

    data_path = file_data + "/DNA_data/"
    data_path = data_path  + data_mode + '/' + data_type + '/'

    path_model = file_data +"/model_data/" + data_mode
    path_model = path_model + method + '/'  + data_type + "/3AUROC/"
    
    print('Data_path:',  data_path)
    print('Model_path:',  path_model)
    print('Embedding_path:',  path_embed)
    
    trainset = mRNA_sequence(type_data='train', data_path_tmp=data_path)
    testset = mRNA_sequence(type_data='test', data_path_tmp=data_path)

    ## STEP 2 build the LSTM network

    # LSTM network parameters
    seq_max_len = 41     # the length of seq
    # n_input = 4          # embedding dimension
    n_hidden = len_seq   # the dimension of the LSTM hidden layers
    n_classes = 2        # Number of categories
    n_layers = 1         # lstm layers

    graph5 = tf.Graph()

    with graph5.as_default():
        # x is input，y is output
        # None is batch_size
        with tf.name_scope('LSTMCNN3_inputs'):
            with tf.variable_scope('LSTMCNN3_in'):
                x = tf.placeholder("float", [None, seq_max_len, n_input], name = 'x_input')
                y = tf.placeholder("float", [None, n_classes], name = 'y_input')
                keep_prob = tf.placeholder(tf.float32, name='keep_prob_input')
                batch_size_ = tf.placeholder(tf.int32, [], name='batch_size_input')

        weights = {
            'in': tf.Variable(tf.random_normal([n_input, n_hidden]),name='w_in'), # (30, 64)
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]),name='w_out') # (64, 2)
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden]), name='b_in'),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b_out')
        }

        def dynamicRNN(x, weights, biases):
            # the shape of x： (batch_size, max_seq_len, n_input)
            X = tf.reshape(x, [-1, n_input])
            X_in = tf.matmul(X, weights['in']) + biases['in']
            X_in = tf.reshape(X_in, [-1, seq_max_len, n_hidden])
            # 定义一个lstm_cell，隐层的大小为n_hidden（之前的参数）
            def lstm_cell():
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                return tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
            
            lstm_cells = []
            for _ in range(n_layers):
                lstm_cells.append(lstm_cell())
            with tf.name_scope('LSTMCNN3_lstm_cells_layers'):
                mlstm_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple = True)
            
            # initiate state
            state = mlstm_cell.zero_state(batch_size_,dtype=tf.float32)
            
            # 使用tf.nn.dynamic_rnn展开时间维度
            # outputs的形状为(batch_size, max_seq_len, n_hidden)
            with tf.variable_scope('LSTMCNN3_outs'):
                outputs33, _ = tf.nn.dynamic_rnn(mlstm_cell, x, initial_state=state, dtype=tf.float32, time_major=False)

            # 只需要最后一个时间维度的输出
            # return tf.matmul(outputs[:,-1,:], weights['out']) + biases['out']
            return outputs33
        
        ## STEP 3 build the CNN network
        
        # initial weights
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev = 0.1)
            return tf.Variable(initial)

        # initial bias
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # convolution layer
        def conv1d(x,f,k):
            return tf.layers.conv1d(x, f, k, padding='VALID')

        # pred is logits
        with tf.name_scope('LSTMCNN3_lstm_layer'):
            lstm_out = dynamicRNN(x, weights, biases)

        # Conv1
        with tf.name_scope('LSTMCNN3_Conv_1'):
            h_conv1 = conv1d(lstm_out, 64, 4)
            dim = h_conv1.get_shape()[1].value*h_conv1.get_shape()[2].value

        # Fc_1
        with tf.name_scope('LSTMCNN3_Fc_1'):
            with tf.name_scope('resh_1'):
                reshape = tf.reshape(h_conv1,[batch_size_, -1])
            with tf.name_scope('w_fc1'):
                W_fc1 = weight_variable([dim, 256])
            with tf.name_scope('b_fc1'):
                b_fc1 = bias_variable([256])
            with tf.name_scope('relu'):
                h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)
            #dropout
            with tf.name_scope('dropout'):
                h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Softmax
        with tf.name_scope('LSTMCNN3_softmax'):
            with tf.name_scope('w_softmax'):
                W_fc3 = weight_variable([256,n_classes])
            with tf.name_scope('b_softmax'):
                b_fc3 = bias_variable([n_classes])
            with tf.name_scope('prediction_softmax'):
                pred1 = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc3) + b_fc3)

        pred2 = tf.matmul(lstm_out[:,21,:], weights['out']) + biases['out']
        pred = pred1+pred2

        # loss
        with tf.name_scope('LSTMCNN3_loss'):
            cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred1, labels=y))
            cost2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred2, labels=y))
            cost3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

            cost = ((cost1+cost2)/2+cost3)/2
            tf.summary.scalar('loss', cost)

        with tf.name_scope('LSTMCNN3_train'):
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # acc
        with tf.name_scope('LSTMCNN3_accuracy'):
            correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('acc', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        saver = tf.train.Saver()

    ## STEP 4 train the network

    if Run_type == 'train':
        # 训练
        with tf.Session(graph = graph5) as sess:
            sess.run(init)
            train_writer = tf.summary.FileWriter(path_model+"logs/train",sess.graph)
            test_writer = tf.summary.FileWriter(path_model+"logs/test",sess.graph)
            step = 1
            while step < training_iters:
                batch_x, batch_y = trainset.data_next(batch_size)
                test_x, test_y = testset.data_next(batch_size)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob:0.5, batch_size_: batch_size})

                # 将数据可视化 10步记录一次
                if step % record_step == 0:
                    summary_train = sess.run(merged, feed_dict={x:batch_x, y:batch_y, keep_prob:1.0, batch_size_:batch_size})
                    summary_test  = sess.run(merged, feed_dict={x:test_x, y:test_y, keep_prob:1.0, batch_size_:batch_size})
                    train_writer.add_summary(summary_train, step)
                    test_writer.add_summary(summary_test, step)

                # 展示 100步一次
                if step % display_step == 0:
                    acc_train = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    acc_test = sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob:1.0, batch_size_:batch_size})
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob:1.0, batch_size_:batch_size})
                    print("%s: Step [%d]  Loss : %.6f, training accuracy :  %.5f, test accuracy :  %.5f" % (datetime.now(), step, loss, acc_train, acc_test))
                    
                step += 1
            print("Optimization Finished!")

            # Save model weights to disk
            saver.save(sess, path_model+"data")

        ## ROC CURVE AND AUROC
        path_pred  = path_model + 'predLSTMCNN3.csv'
        path_label = path_model + 'labelLSTMCNN3.csv'

        with tf.Session(graph = graph5) as sess:
            sess.run(init)
            saver.restore(sess, path_model + "data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            pred_neg = []
            for i in range(10):
                test_data, test_label = testset.data_next(num_pos, rate_pos=0)
                pred_ = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
                pred_neg.extend(pred_)
                
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            # 记录预测值和真实值
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)


        # Draw the ROC curve and get the AUC
        auroc = ROC(path_pred, path_label, iter=iter_size)
        print(auroc)
    
    elif Run_type == 'test':
        
        testset = mRNA_sequence(type_data='test', data_path_tmp="")

        path_pred  = '/predLSTMCNN3.csv'
        path_label = '/labelLSTMCNN3.csv'
 
        # 运行得到预测值和真实值
        with tf.Session(graph = graph5) as sess:
            sess.run(init)
            saver.restore(sess, test_path + "/3AUROC/model_data/data")
            num_neg = testset.num_neg
            num_pos = testset.num_pos
            print("num_neg:",num_neg,'  num_pos:',num_pos)

            test_data, test_label = testset.data_next(num_pos, rate_pos=1)
            pred_pos = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_pos})
            pred_pos = list(pred_pos)

            test_data, test_label = testset.data_next(num_neg, rate_pos=0)
            pred_neg = sess.run(pred, feed_dict={x: test_data, y: test_label, keep_prob:1.0, batch_size_:num_neg})
            pred_neg = list(pred_neg)
                
            y_pos = [[1.0,0.]]*len(pred_pos)
            y_neg = [[0.,1.0]]*len(pred_neg)
            
            with open(path_pred, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(pred_pos)
                writer.writerows(pred_neg)

            with open(path_label, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(y_pos)
                writer.writerows(y_neg)

        auroc = ROC(path_pred, path_label, iter=iter_size)