from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint as pp
import csv
import os
import time

tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
y_data = ["1", "2", "3", "4", "5"]

#print(x_data.shape, y_data.shape)

nb_classes = 5  # 1:punch_l 2:punch_r 3:punch_l2 4:punch_r2 5:hold

X = tf.placeholder(tf.float32, [None, 864])
Y = tf.placeholder(tf.int32, [None, 1])  # 1:punch_l 2:punch_r 3:punch_l2 4:punch_r2 5:hold
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 1])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([864, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                 labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.AdamOptimizer(learning_rate=0.000001).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

def python_init():
	#config = tf.ConfigProto()
	#config.gpu_options.per_process_gpu_memory_fraction = 0.3
	global sess
	sess.run(init)

	saver = tf.train.Saver()
	saver.restore(sess, os.getcwd() + "/model-softmax.ckpt")
	print(os.getcwd() + "/model-softmax.ckpt")

'''def action_classification(arg):
	data = np.array(arg, dtype=np.float)
	print(data)
	data = data.reshape(1, 16, 54)
	global sess
	global prediction
	global pre_data
	#predict_output = sess.run(prediction, feed_dict={X: data})
	#print("Prediction:", predict_output)
	return predict_output'''

def action_classification(arg):
	global sess
	global prediction
	dataX = [[0 for i in range(864)]]
	line = arg.split(',')
	linetodata = list(line)
	c = 0
	for a in range(864):
		dataX[0][a] = linetodata[c]
		c = c + 1
	data = np.array(dataX, dtype=np.float32)
	predict_output = sess.run(prediction, feed_dict={X: data})
	for p, y in zip(predict_output, y_data):
		if(p == 0) :
			print("Left Punch 1")
		elif(p == 1) :
			print("Right Punch 1")
		elif(p == 2) :
			print("Left Punch 2")
		elif(p == 3) :
			print("Right Punch 2")
		elif(p == 4) :
			print("Hold")
#	    print("Prediction: {}".format(p))


#def action_classification(arg):
#	print("Enter")
#	global sess
#	global prediction
#        
#	dataX = [[[0 for rows in range(54)]for cols in range(16)]]
#	c = 0;
#	strline = hello.readline()
#	line = strline.split(',')
#	linetodata = list(line)
#	print(linetodata)
#	for a in range(16):
#		for b in range(54):
#			dataX[0][a][b] = linetodata[c]
#			c = c + 1
#	data = np.array(dataX, dtype=np.float32)
#	predict_output = sess.run(prediction, feed_dict={X: pre_data})
#	print("Prediction:", predict_output)
#	return predict_output

def python_close():
	global sess
	sess.close()


#pp.pprint(dataX)

#pp.pprint(tf.shape(x_data))

#cell = rnn.BasicLSTMCell(num_units=2, state_is_tuple=True)
#outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
#sess = tf.InteractiveSession()
#sess.run(tf.global_variables_initializer())
#pp.pprint(outputs.eval())
