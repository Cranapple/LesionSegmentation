# Starting with 5 covolution layers, 2 fully connected and a softmax. Kernals are 3x3. 2D convolutions only for now

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']

high_patch_size = 25
output_size = 9
#num_labels = 10
num_channels = 1 # grayscale

#split dataset into validation and test as well. Scramble images for 2D.

#Probably need to reformat patches to float.

#Create new accuracy function

batch_size = 10
#patch_size = 5
depth1 = 30
depth1 = 40
depth1 = 50
num_hidden = 150

kernal_size = 3

graph = tf.Graph()

with graph.as_default():

	# Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, high_patch_size, high_patch_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size, output_size))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	
	# Variables.
	cov1_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, num_channels, depth1], stddev=0.1))
	cov1_biases = tf.Variable(tf.zeros([depth1]))
	cov2_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, depth1, depth2], stddev=0.1))
	cov2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
	cov3_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, depth2, depth2], stddev=0.1))
	cov3_biases = tf.Variable(tf.zeros([depth]))
	cov4_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, depth2, depth3], stddev=0.1))
	cov4_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
	cov5_weights = tf.Variable(tf.truncated_normal(
			[patch_size, patch_size, depth3, num_hidden], stddev=0.1))
	cov5_biases = tf.Variable(tf.zeros([depth]))

	full1_weights = tf.Variable(tf.truncated_normal(
			[image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
	full1_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
	full2_weights = tf.Variable(tf.truncated_normal(
			[num_hidden, num_labels], stddev=0.1))
	full2_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	
	class_weights = tf.Variable(tf.truncated_normal(
			[num_hidden, num_labels], stddev=0.1))
	class_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	# Model.
	def model(data):
		conv = tf.nn.conv2d(data, layer1_weights, [1, kernal_size, kernal_size, 1], padding='SAME')
		hidden = tf.nn.relu(conv + layer1_biases)
		conv = tf.nn.conv2d(hidden, layer2_weights, [1, kernal_size, kernal_size, 1], padding='SAME')
		hidden = tf.nn.relu(conv + layer2_biases)
		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
		return tf.matmul(hidden, layer4_weights) + layer4_biases
	
	# Training computation.
	logits = model(tf_train_dataset)
	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
		
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))


num_steps = 1001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 50 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(
				valid_prediction.eval(), valid_labels))
	print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))