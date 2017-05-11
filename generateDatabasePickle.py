from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
from parameters import *
import random
import time

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']

#features: size, patch_size, patch_size, 1
#labels: size, patch_size, patch_size, 2. First is lesion, second is non-lesion

patch_features = np.zeros((database_size, patch_size, patch_size, 1), dtype=np.float32)
patch_labels = np.zeros((database_size, output_size, output_size, 2), dtype=np.float32)

patchSpan = patch_size // 2
outputSpan = output_size // 2
numLesions = 0
random.seed(time.time())
for n in range(database_size):
	success = False
	while(not success):
		i = random.randrange(len(features))
		s = random.randrange(len(features[i]))
		x = random.randrange(len(features[i][s]))
		y = random.randrange(len(features[i][s][x]))
		#for xi in range(x-patchSpan, x+patchSpan+1):
		#	for yi in range(y-patchSpan, y+patchSpan+1):
		#		if xi < 0 or xi >= len(features[i][s]) or yi < 0 or yi >= len(features[i][s][x]): 
		#			patch_features[n, xi-x+patchSpan, yi-y+patchSpan, 0] = 0
		#		else:
		#			patch_features[n, xi-x+patchSpan, yi-y+patchSpan, 0] = features[i][s][xi, yi]
		#for xi in range(x-outputSpan, x+outputSpan+1):
		#	for yi in range(y-outputSpan, y+outputSpan+1):
		#		if xi < 0 or xi >= len(labels[i][s]) or yi < 0 or yi >= len(labels[i][s][x]): 
		#			patch_labels[n, xi-x+outputSpan, yi-y+outputSpan, 0] = 0;
		#		else:
		#			patch_labels[n, xi-x+outputSpan, yi-y+outputSpan, 0] = labels[i][s][xi, yi]
		try:
			patch_features[n, :, :, 0] = features[i][s][x-patchSpan:x+patchSpan+1, y-patchSpan:y+patchSpan+1]
			patch_labels[n, :, :, 0] = labels[i][s][x-outputSpan:x+outputSpan+1, y-outputSpan:y+outputSpan+1]
		except:
			continue
		patch_labels[n, :, :, 1] = abs(1 - patch_labels[n, :, :, 0])

		if np.sum(patch_labels[n, :, :, 0]) > 0.5:
			numLesions += 1
			success = True
			#print(np.sum(patch_labels[n, :, :, 0]))

		if n // 2 <= numLesions:
			success = True;

#Add in centering and normalization

train_features, valid_features = np.split(patch_features, [train_size])
train_labels, valid_labels = np.split(patch_labels, [train_size])
pickle_file = 'lesionDatabase.pickle'

print("Number of samples with lesions: ", numLesions)
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_features': train_features,
    'train_labels': train_labels,
    'valid_features': valid_features,
    'valid_labels': valid_labels
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
