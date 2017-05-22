import os
import numpy as np
from matplotlib import pyplot
from six.moves import cPickle as pickle
from parameters import *
import random

#Program to run and vizualize a segmentation on a image using the saved model

#Testing the dataset

pickle_file = 'lesionDatabase.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_features = save['train_features']
	train_labels = save['train_labels']
	valid_features = save['valid_features']
	valid_labels = save['valid_labels']

#print(train_features.shape)
#print(train_labels.shape)
#print(valid_features.shape)
#print(valid_labels.shape)

for i in range(0,10):#random.sample(range(0, 10), 10):
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

#print(train_features.shape)
#print(train_labels.shape)
#print(valid_features.shape)
#print(valid_labels.shape)

for i in random.sample(range(0, 13), 0):
	for z in random.sample(range(0, 20), 0):
		img = features[i][z]
		pyplot.imshow(img)
		pyplot.show()
		img = labels[i][z]
		pyplot.imshow(img)
		pyplot.show()


#Testing the model

#To be continued!!!!