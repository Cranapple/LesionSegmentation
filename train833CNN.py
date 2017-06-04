
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from parameters import *
import sys
from math import sqrt
import os

modelName = "833CNN"
pickle_file = 'lesionDatabase.pickle'
saveInterval = 1000
kernel_size = 3

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
	tf_valid_labels = tf.constant(valid_labels, name="validLabels")

	tf_test_features = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 1), name="features")

	# Variables.
	cov1_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, depth1], stddev=sqrt(2.0/depth1)))
	cov2_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth1, depth1], stddev=sqrt(2.0/depth1)))
	cov3_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth1, depth2], stddev=sqrt(2.0/depth1)))
	cov4_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth2], stddev=sqrt(2.0/depth2)))
	cov5_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth2], stddev=sqrt(2.0/depth2)))
	cov6_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth3], stddev=sqrt(2.0/depth3)))
	cov7_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth3, depth3], stddev=sqrt(2.0/depth3)))
	cov8_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth3, depth3], stddev=sqrt(2.0/depth3)))

	cov1_biases = tf.Variable(tf.zeros([depth1]))
	cov2_biases = tf.Variable(tf.zeros([depth1]))
	cov3_biases = tf.Variable(tf.zeros([depth2]))
	cov4_biases = tf.Variable(tf.zeros([depth2]))
	cov5_biases = tf.Variable(tf.zeros([depth2]))
	cov6_biases = tf.Variable(tf.zeros([depth3]))
	cov7_biases = tf.Variable(tf.zeros([depth3]))
	cov8_biases = tf.Variable(tf.zeros([depth3]))

	class_weights = tf.Variable(tf.truncated_normal([1, 1, depth3, 2], stddev=sqrt(2.0/depth3))) #First for lesion, second for non-lesion
	class_biases = tf.Variable(tf.zeros([2]))

	def tf_DSC(predictions, labels):
		s = tf.shape(predictions)[0:3]
		ones = tf.ones(s, tf.int64)
		zeros = tf.zeros(s, tf.int64)
		TP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(predictions, 3), ones), tf.equal(tf.argmax(labels, 3), ones)), tf.float32))
		FP = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(tf.argmax(predictions, 3), ones), tf.equal(tf.argmax(labels, 3), zeros)), tf.float32))
		P = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(labels, 3), ones), tf.float32))
		return 100.0 * TP / (FP + P + epsilon)

	def tf_accuracy(predictions, labels):
		s = tf.cast(tf.shape(predictions), tf.float32)
		return 100.0 * tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predictions, 3), tf.argmax(labels, 3)), tf.float32)) / (s[0] * s[1] * s[2])

	# Model.
	def model(data, isTraining=True):
		conv = tf.nn.conv2d(data, cov1_weights, [1, 1, 1, 1], padding='VALID')				#cov1
		hidden = tf.nn.relu(conv + cov1_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov2_weights, [1, 1, 1, 1], padding='VALID')			#cov2
		hidden = tf.nn.relu(conv + cov2_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov3_weights, [1, 1, 1, 1], padding='VALID')			#cov3
		hidden = tf.nn.relu(conv + cov3_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov4_weights, [1, 1, 1, 1], padding='VALID')			#cov4
		hidden = tf.nn.relu(conv + cov4_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov5_weights, [1, 1, 1, 1], padding='VALID')				#cov5
		hidden = tf.nn.relu(conv + cov5_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov6_weights, [1, 1, 1, 1], padding='VALID')			#cov6
		hidden = tf.nn.relu(conv + cov6_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov7_weights, [1, 1, 1, 1], padding='VALID')			#cov7
		hidden = tf.nn.relu(conv + cov7_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, cov8_weights, [1, 1, 1, 1], padding='VALID')			#cov8
		hidden = tf.nn.relu(conv + cov8_biases)
		if(isTraining):
			hidden = tf.nn.dropout(hidden, dropoutRate)

		conv = tf.nn.conv2d(hidden, class_weights, [1, 1, 1, 1], padding='VALID')			#Classification
		return conv + class_biases
	
	# Training computation.
	logits = model(tf_train_features)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(logits, [-1, 2]), labels=tf.reshape(tf_train_labels, [-1, 2]))
			+ l2Rate*tf.nn.l2_loss(cov1_weights)
			+ l2Rate*tf.nn.l2_loss(cov2_weights)
			+ l2Rate*tf.nn.l2_loss(cov3_weights)
			+ l2Rate*tf.nn.l2_loss(cov4_weights)
			+ l2Rate*tf.nn.l2_loss(cov5_weights)
			+ l2Rate*tf.nn.l2_loss(cov6_weights)
			+ l2Rate*tf.nn.l2_loss(cov7_weights)
			+ l2Rate*tf.nn.l2_loss(cov8_weights)
			+ l2Rate*tf.nn.l2_loss(cov1_biases)
			+ l2Rate*tf.nn.l2_loss(cov2_biases)
			+ l2Rate*tf.nn.l2_loss(cov3_biases)
			+ l2Rate*tf.nn.l2_loss(cov4_biases)
			+ l2Rate*tf.nn.l2_loss(cov5_biases)
			+ l2Rate*tf.nn.l2_loss(cov6_biases)
			+ l2Rate*tf.nn.l2_loss(cov7_biases)
			+ l2Rate*tf.nn.l2_loss(cov8_biases)
			, name="loss")
	# Optimizer.
	optimizer = tf.train.AdamOptimizer(0.005).minimize(loss, name="optimizer")

	# Predictions
	train_prediction = tf.nn.softmax(logits, name="trainPred")
	valid_prediction = tf.nn.softmax(model(tf_valid_features), name="validPred")
	test_prediction = tf.nn.softmax(model(tf_test_features, isTraining=False), name="labels")

	#Accuracies
	trainAccuracy = tf_accuracy(train_prediction, tf_train_labels)
	trainDSC = tf_DSC(train_prediction, tf_train_labels)
	validAccuracy = tf_accuracy(valid_prediction, tf_valid_labels)
	validDSC = tf_DSC(valid_prediction, tf_valid_labels)

	lossSum = tf.summary.scalar("loss", loss)
	trainAccSum = tf.summary.scalar("Train Accuracy", trainAccuracy)
	trainDSCSum = tf.summary.scalar("Train DSC", trainDSC)
	validAccSum = tf.summary.scalar("Valid Accuracy", validAccuracy)
	validDSCSum = tf.summary.scalar("Valid DSC", validDSC)

	validSum = tf.summary.merge_all()
	trainSum = tf.summary.merge((lossSum, trainAccSum, trainDSCSum))

#Save the model for running tests on images

num_steps = 10001

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	saver = tf.train.Saver(max_to_keep=10)
	if not os.path.exists('./lesion_tensorboard/' + modelName):
		os.makedirs('./lesion_tensorboard/' + modelName)
	writer = tf.summary.FileWriter('./lesion_tensorboard/' + modelName, session.graph)
	print('Valid Label Percent: %.1f%%\n' % (100 * percentLesion(valid_labels)))
	sys.stdout.flush()
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_features = train_features[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :, :, :]
		feed_dict = {tf_train_features : batch_features, tf_train_labels : batch_labels}
		if (step % 100 == 0):
			_, summary = session.run([optimizer, validSum], feed_dict=feed_dict)
			writer.add_summary(summary, step)
		elif (step % 10 == 0):
			_, summary = session.run([optimizer, trainSum], feed_dict=feed_dict)
			writer.add_summary(summary, step)
		else:
			_, summary = session.run([optimizer, lossSum], feed_dict=feed_dict)
			writer.add_summary(summary, step)

		print(step)
		sys.stdout.flush()
		#_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		#if (step % 5 == 0):
		#	print('Minibatch loss at step %d: %f' % (step, l))
		#	print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
		#	print('Minibatch DSC: %.1f%%' % DSC(predictions, batch_labels))
		#	if (step % 100 == 0):
		#		print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
		#		print('Validation DSC: %.1f%%' % DSC(valid_prediction.eval(), valid_labels))
		#	print('\n')
		#	sys.stdout.flush()
		if (step % saveInterval == 0):
			if not os.path.exists('./lesion_models/' + modelName):
				os.makedirs('./lesion_models/' + modelName)
			saver.save(session, './lesion_models/' + modelName + "/" + modelName, global_step=step)
