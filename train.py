# Starting with 5 covolution layers, 2 fully connected and a softmax. Kernals are 3x3. 2D convolutions only for now

#TODOs
#Fix initilization weights -> approximated with truncated normal. What to set biases?

#Function to format and create test + validation. Need to also change values to float
#SEPERATE FILE, new pickle

#Normalize pixel values? Also normalize cov outputs? Batch normalization?

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range

def accuracy(predictions, labels):
	  return (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) / predictions.shape[0])

pickle_file = 'lesionDataset.pickle'

#Labels are of [datasetSize, patch_size, patch_size, 1]
#Features are of [datasetSize, patch_size, patch_size, 2]
with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_features = save['train_features']
	train_labels = save['train_labels']
	valid_features = save['valid_features']
	valid_labels = save['valid_labels']

patch_size = 25
output_size = 25 - 10 #Change depending on number of conv.
batch_size = 10
depth1 = 30
depth2 = 40
depth3 = 50
num_hidden = 150
kernel_size = 3

graph = tf.Graph()

with graph.as_default():

	# Input data.
	tf_train_features = tf.placeholder(tf.float32, shape=(batch_size, patch_size, patch_size, 1))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, output_size, output_size, 2))
	tf_valid_features = tf.constant(valid_features)
	tf_valid_labels = tf.constant(valid_labels)

	# Variables.
	cov1_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, 1, depth1], stddev=0.1))
	cov1_biases = tf.Variable(tf.zeros([depth1]))

	cov2_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth1, depth2], stddev=0.1))
	cov2_biases = tf.Variable(tf.zeros([depth2]))

	cov3_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth2], stddev=0.1))
	cov3_biases = tf.Variable(tf.zeros([depth2]))

	cov4_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth2, depth3], stddev=0.1))
	cov4_biases = tf.Variable(tf.zeros([depth3]))

	cov5_weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, depth3, depth3], stddev=0.1))
	cov5_biases = tf.Variable(tf.zeros([depth3]))


	full1_weights = tf.Variable(tf.truncated_normal([1, 1, depth3, num_hidden], stddev=0.1))
	full1_biases = tf.Variable(tf.zeros([num_hidden]))

	full2_weights = tf.Variable(tf.truncated_normal([1, 1, num_hidden, num_hidden], stddev=0.1))
	full2_biases = tf.Variable(tf.zeros([num_hidden]))


	class_weights = tf.Variable(tf.truncated_normal([1, 1, num_hidden, 2], stddev=0.1)) #First for lesion, second for non-lesion
	class_biases = tf.Variable(tf.zeros([2]))

	# Model.
	def model(data):
		conv = tf.nn.conv2d(data, cov1_weights, [1, 1, 1, 1], padding='VALID')				#cov1
		hidden = tf.nn.relu(conv + cov1_biases)
		conv = tf.nn.conv2d(hidden, cov2_weights, [1, 1, 1, 1], padding='VALID')			#cov2
		hidden = tf.nn.relu(conv + cov2_biases)
		conv = tf.nn.conv2d(data, cov3_weights, [1, 1, 1, 1], padding='VALID')				#cov3
		hidden = tf.nn.relu(conv + cov3_biases)
		conv = tf.nn.conv2d(hidden, cov4_weights, [1, 1, 1, 1], padding='VALID')			#cov4
		hidden = tf.nn.relu(conv + cov4_biases)
		conv = tf.nn.conv2d(data, cov5_weights, [1, 1, 1, 1], padding='VALID')				#cov5
		hidden = tf.nn.relu(conv + cov5_biases)
		conv = tf.nn.conv2d(hidden, full1_weights, [1, 1, 1, 1], padding='VALID')			#FC1
		hidden = tf.nn.relu(conv + full1_biases)
		conv = tf.nn.conv2d(data, full2_weights, [1, 1, 1, 1], padding='VALID')				#FC2
		hidden = tf.nn.relu(conv + full2_biases)
		conv = tf.nn.conv2d(hidden, class_weights, [1, 1, 1, 1], padding='VALID')			#Classification
		return conv + class_biases
	
	# Training computation.
	logits = model(train_features)
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.reshape(logits, [-1, 2]), logits=tf.reshape(train_labels, [-1, 2])))
		
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)	#Fiddle this parameter
	
	# Predictions
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(valid_features))
	#Test dataset later?

#Save the model for running tests on images

num_steps = 1001

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print('Initialized')
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :, :, :]
		feed_dict = {tf_train_features : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 50 == 0):
			print('Minibatch loss at step %d: %f' % (step, l))
			print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
			print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
	#print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
