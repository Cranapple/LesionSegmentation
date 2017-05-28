import os
import numpy as np
from matplotlib import pyplot
from six.moves import cPickle as pickle
from parameters import *
import random
import tensorflow as tf

#Program to run and vizualize a segmentation on a image using the saved model

numPatchSamples = 0
numImgSamples = 0
numTestSamples = 3

#Testing the dataset

pickle_file = 'lesionDatabase.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_features = save['train_features']
	train_labels = save['train_labels']
	valid_features = save['valid_features']
	valid_labels = save['valid_labels']

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']

for i in random.sample(range(0, 10), numPatchSamples):
	img1 = valid_features[i, :, :, 0]
	img2 = valid_labels[i, :, :, 0]
	img3 = valid_labels[i, :, :, 1]
	img = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
	img[:, :, 0] = img1
	img[8:17, 8:17, 1] = img2
	img[8:17, 8:17, 2] = img3
	pyplot.imshow(img)
	pyplot.show()

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']


for i in random.sample(range(0, 13), numImgSamples):
	for z in random.sample(range(0, len(features[i])-1), 1):
		img = features[i][z]
		pyplot.imshow(img)
		pyplot.show()
		img = labels[i][z]
		pyplot.imshow(img)
		pyplot.show()


#Testing the model
modelName = "lesion_model"
step = 1500

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
	new_saver = tf.train.import_meta_graph("./" + modelName + "-" + str(step) + ".meta")
	new_saver.restore(session, tf.train.latest_checkpoint('./'))
	session.run(tf.global_variables_initializer())
	#print(len(features))
	for i in random.sample(range(0, 13), numTestSamples):
		print(i)
		for z in random.sample(range(0, len(features[i])), 1):
			imgF = features[i][z]
			imgL = labels[i][z]
			imgPL = np.zeros(imgL.shape, dtype=np.float32)
			xDim = (imgF.shape[0] - patch_size) // output_size
			yDim = (imgF.shape[1] - patch_size) // output_size
			patches = np.zeros((xDim*yDim, patch_size, patch_size, 1), dtype=np.float32)
			#outputs = np.zeros((xDim*yDim, output_size, output_size, 2), dtype=np.float32)
			for x in range(xDim):
				for y in range(yDim):
					patches[x * yDim + y, :, :, 0] = imgF[x*output_size:x*output_size+patch_size, y*output_size:y*output_size+patch_size]

			graph = tf.get_default_graph()
			tf_test_features = graph.get_tensor_by_name("features:0")
			test_prediction = graph.get_tensor_by_name("labels:0")
			#print(patches.shape)
			#print(imgPL.shape)
			feed_dict = {tf_test_features : patches}
			outputs = session.run([test_prediction], feed_dict=feed_dict)
			outputs = np.array(outputs).reshape((-1, output_size, output_size, 2))
			for x in range(xDim):
				for y in range(yDim):
					xP = x*output_size + (patch_size - output_size) // 2
					yP = y*output_size + (patch_size - output_size) // 2
					#print(outputs.shape)
					imgPL[xP:xP+output_size, yP:yP+output_size] = np.argmin(outputs[x*yDim + y], 2)#outputs[x*yDim + y, :, :, 0]
			pyplot.subplot(131)
			pyplot.imshow(imgF)
			pyplot.subplot(132)
			pyplot.imshow(imgL)
			pyplot.subplot(133)
			pyplot.imshow(imgPL)
			pyplot.show()
