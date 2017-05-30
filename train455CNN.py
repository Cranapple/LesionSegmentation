
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from parameters import *
import sys
from math import sqrt
import os

modelName = "455CNN"
pickle_file = 'lesionDatabase.pickle'
saveInterval = 1000

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
	tf_train_features = tf.placeholder(tf.float32, shape=(batch_size, patch_size, patch_size, 1), name="trainFeatures")
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size, output_size, 2), name="trainLabels")
	tf_valid_features = tf.constant(valid_features, name="validFeatures")
	#tf_valid_labels = tf.constant(valid_labels)

	tf_test_features = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 1), name="features")

	# Variables.
	cov1_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, depth1], stddev=sqrt(2.0/depth1)))
	cov2_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth1, depth2], stddev=sqrt(2.0/depth1)))
	cov3_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth2], stddev=sqrt(2.0/depth2)))
	cov4_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth3], stddev=sqrt(2.0/depth2)))

	cov1_biases = tf.Variable(tf.zeros([depth1]))
	cov2_biases = tf.Variable(tf.zeros([depth2]))
	cov3_biases = tf.Variable(tf.zeros([depth2]))
	cov4_biases = tf.Variable(tf.zeros([depth3]))

	class_weights = tf.Variable(tf.truncated_normal([1, 1, depth3, 2], stddev=sqrt(2.0/depth3))) #First for lesion, second for non-lesion
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
		#convNorm = batch_norm_wrapper(conv, isTraining)
		#hidden = tf.nn.relu(convNorm)
		hidden = tf.nn.relu(conv + cov1_biases)
		conv = tf.nn.conv2d(hidden, cov2_weights, [1, 1, 1, 1], padding='VALID')			#cov2
		#convNorm = batch_norm_wrapper(conv, isTraining)
		#hidden = tf.nn.relu(convNorm)
		hidden = tf.nn.relu(conv + cov2_biases)
		conv = tf.nn.conv2d(hidden, cov3_weights, [1, 1, 1, 1], padding='VALID')			#cov3
		#convNorm = batch_norm_wrapper(conv, isTraining)
		#hidden = tf.nn.relu(convNorm)
		hidden = tf.nn.relu(conv + cov3_biases)
		conv = tf.nn.conv2d(hidden, cov4_weights, [1, 1, 1, 1], padding='VALID')			#cov4
		#convNorm = batch_norm_wrapper(conv, isTraining)
		#hidden = tf.nn.relu(convNorm)
		hidden = tf.nn.relu(conv + cov4_biases)

		conv = tf.nn.conv2d(hidden, class_weights, [1, 1, 1, 1], padding='VALID')			#Classification
		return conv + class_biases
	
	# Training computation.
	logits = model(tf_train_features)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(logits, [-1, 2]), labels=tf.reshape(tf_train_labels, [-1, 2])), name="loss")
	# Optimizer.
	optimizer = tf.train.AdamOptimizer(0.005).minimize(loss, name="optimizer")	#Fiddle this parameter
	#optimizer = tf.train.MomentumOptimizer(0.0005, 0.95, use_nesterov=True).minimize(loss, name="optimizer")	#Fiddle this parameter

	# Predictions
	train_prediction = tf.nn.softmax(logits, name="trainPred")
	valid_prediction = tf.nn.softmax(model(tf_valid_features), name="validPred")
	test_prediction = tf.nn.softmax(model(tf_test_features, isTraining=False), name="labels")

#Save the model for running tests on images

num_steps = 20001

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
		if (step % 5 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Minibatch DSC: %.1f%%' % DSC(predictions, batch_labels))
			if (step % 100 == 0):
				print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
				print('Validation DSC: %.1f%%' % DSC(valid_prediction.eval(), valid_labels))
			print('\n')
			sys.stdout.flush()
		if (step % saveInterval == 0):
			if not os.path.exists('.\lesion_models\\' + modelName):
				os.makedirs('.\lesion_models\\' + modelName)
			saver.save(session, '.\lesion_models\\' + modelName + "\\" + modelName, global_step=step)
