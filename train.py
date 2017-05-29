
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from parameters import *
import sys
from math import sqrt

epsilon = 1e-3

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) / (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))

def percentLesion(labels):
	shape = labels.shape
	sum = shape[0] * shape[1] * shape[2]
	#print(np.argmax(labels, 3))
	#print(predictions)
	return (sum - np.sum(np.argmax(labels, 3))) / sum

pickle_file = 'lesionDatabase.pickle'

#Labels are of [datasetSize, output_size, output_size, 1]
#Features are of [datasetSize, patch_size, patch_size, 2]
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_features = save['train_features']
	train_labels = save['train_labels']
	valid_features = save['valid_features']
	valid_labels = save['valid_labels']

device_name = "gpu"

if device_name == "gpu":
	device_name = "/GPU:0"
else:
	device_name = "/cpu:0"

#graph = tf.Graph()

#with graph.as_default():
with tf.device(device_name):
	# Input data.
	tf_train_features = tf.placeholder(tf.float32, shape=(batch_size, patch_size, patch_size, 1))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size, output_size, 2))
	tf_valid_features = tf.constant(valid_features)
	#tf_valid_labels = tf.constant(valid_labels)

	tf_test_features = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 1), name="features")

	# Variables.
	cov1_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, depth1], stddev=sqrt(2.0/depth1)))
	cov1_biases = tf.Variable(tf.zeros([depth1]))
	#cov1_norm_weights = tf.Variable(tf.ones([depth1]))
	#cov1_norm_biases = tf.Variable(tf.zeros([depth1]))

	cov2_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth1, depth2], stddev=sqrt(2.0/depth1)))
	cov2_biases = tf.Variable(tf.zeros([depth2]))
	#cov2_norm_weights = tf.Variable(tf.ones([depth2]))
	#cov2_norm_biases = tf.Variable(tf.zeros([depth2]))

	cov3_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth2], stddev=sqrt(2.0/depth2)))
	cov3_biases = tf.Variable(tf.zeros([depth2]))
	#cov3_norm_weights = tf.Variable(tf.ones([depth2]))
	#cov3_norm_biases = tf.Variable(tf.zeros([depth2]))

	cov4_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth3], stddev=sqrt(2.0/depth2)))
	cov4_biases = tf.Variable(tf.zeros([depth3]))
	#cov4_norm_weights = tf.Variable(tf.ones([depth3]))
	#cov4_norm_biases = tf.Variable(tf.zeros([depth3]))

	#cov5_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth3, depth3], stddev=0.01))
	#cov5_biases = tf.Variable(tf.zeros([depth3]))

	full1_weights = tf.Variable(tf.truncated_normal([1, 1, depth3, num_hidden], stddev=sqrt(2.0/depth3)))
	full1_biases = tf.Variable(tf.zeros([num_hidden]))
	#full1_norm_weights = tf.Variable(tf.ones([num_hidden]))
	#full1_norm_biases = tf.Variable(tf.zeros([num_hidden]))

	full2_weights = tf.Variable(tf.truncated_normal([1, 1, num_hidden, num_hidden], stddev=sqrt(2.0/num_hidden)))
	full2_biases = tf.Variable(tf.zeros([num_hidden]))
	#full2_norm_weights = tf.Variable(tf.ones([num_hidden]))
	#full2_norm_biases = tf.Variable(tf.zeros([num_hidden]))


	class_weights = tf.Variable(tf.truncated_normal([1, 1, num_hidden, 2], stddev=sqrt(2.0/num_hidden))) #First for lesion, second for non-lesion
	class_biases = tf.Variable(tf.zeros([2]))

	def batch_norm_wrapper(inputs, is_training, decay = 0.995):

		scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
		beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
		pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
		pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

		if is_training:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
			train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
			train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
			with tf.control_dependencies([train_mean, train_var]):
				return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
		else:
			return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

	# Model.
	def model(data, isTraining=True):
		conv = tf.nn.conv2d(data, cov1_weights, [1, 1, 1, 1], padding='VALID')				#cov1
		#mean, var = tf.nn.moments(conv,[0,1,2])
		#convNorm = tf.nn.batch_normalization(conv,mean,var,cov1_norm_biases,cov1_norm_weights,epsilon)
		convNorm = batch_norm_wrapper(conv, isTraining)
		hidden = tf.nn.relu(convNorm + cov1_biases)
		conv = tf.nn.conv2d(hidden, cov2_weights, [1, 1, 1, 1], padding='VALID')			#cov2
		#mean, var = tf.nn.moments(conv,[0,1,2])
		#convNorm = tf.nn.batch_normalization(conv,mean,var,cov2_norm_biases,cov2_norm_weights,epsilon)
		convNorm = batch_norm_wrapper(conv, isTraining)
		hidden = tf.nn.relu(convNorm + cov2_biases)
		conv = tf.nn.conv2d(hidden, cov3_weights, [1, 1, 1, 1], padding='VALID')			#cov3
		#mean, var = tf.nn.moments(conv,[0,1,2])
		#convNorm = tf.nn.batch_normalization(conv,mean,var,cov3_norm_biases,cov3_norm_weights,epsilon)
		convNorm = batch_norm_wrapper(conv, isTraining)
		hidden = tf.nn.relu(convNorm + cov3_biases)
		conv = tf.nn.conv2d(hidden, cov4_weights, [1, 1, 1, 1], padding='VALID')			#cov4
		#mean, var = tf.nn.moments(conv,[0,1,2])
		#convNorm = tf.nn.batch_normalization(conv,mean,var,cov4_norm_biases,cov4_norm_weights,epsilon)
		convNorm = batch_norm_wrapper(conv, isTraining)
		hidden = tf.nn.relu(convNorm + cov4_biases)

		conv = tf.nn.conv2d(hidden, full1_weights, [1, 1, 1, 1], padding='VALID')			#FC1
		#mean, var = tf.nn.moments(conv,[0,1,2])
		#convNorm = tf.nn.batch_normalization(conv,mean,var,full1_norm_biases,full1_norm_weights,epsilon)
		convNorm = batch_norm_wrapper(conv, isTraining)
		hidden = tf.nn.relu(convNorm + full1_biases)
		conv = tf.nn.conv2d(hidden, full2_weights, [1, 1, 1, 1], padding='VALID')			#FC2
		#mean, var = tf.nn.moments(conv,[0,1,2])
		#convNorm = tf.nn.batch_normalization(conv,mean,var,full1_norm_biases,full2_norm_weights,epsilon)
		convNorm = batch_norm_wrapper(conv, isTraining)
		hidden = tf.nn.relu(convNorm + full2_biases)
		conv = tf.nn.conv2d(hidden, class_weights, [1, 1, 1, 1], padding='VALID')			#Classification
		return conv + class_biases
	
	# Training computation.
	logits = model(tf_train_features)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(logits, [-1, 2]), labels=tf.reshape(tf_train_labels, [-1, 2])))
	#test = tf.reshape(tf_train_labels, [-1, 2])
	# Optimizer.
	optimizer = tf.train.AdamOptimizer(0.005).minimize(loss)	#Fiddle this parameter
	
	# Predictions
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_features))
	test_prediction = tf.nn.softmax(model(tf_test_features, isTraining=False), name="labels")
	#Test dataset later?

#Save the model for running tests on images

num_steps = 10000000

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	saver = tf.train.Saver()
	print('Valid Label Percent: %.1f%%\n' % (100 * percentLesion(valid_labels)))
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_features = train_features[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :, :, :]
		feed_dict = {tf_train_features : batch_features, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		#print(predictions.shape)
		if (step % 1 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			#print('Prediction Percent: %.1f%%' % (100 * percentLesion(predictions)))
			#print('Batch Percent: %.1f%%' % (100 * percentLesion(batch_labels)))
			if (step % 10 == 0):
				print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
				#print('Valid Pred Percent: %.1f%%' % (100 * percentLesion(valid_prediction.eval())))
				#print('Valid Label Percent: %.1f%%' % (100 * percentLesion(valid_labels)))
			print('\n')
			sys.stdout.flush()
		if (step % 250 == 0):
			saver.save(session, '.\lesion_model', global_step=step)
	#print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
