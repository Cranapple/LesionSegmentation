from __future__ import print_function
import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from parameters import *
import random
import time

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = []
    shuffled_b = []
    permutation = np.random.permutation(len(a))
    for old_index in permutation:
        shuffled_a.append(a[old_index])
        shuffled_b.append(b[old_index])
    return shuffled_a, shuffled_b

def shuffle_in_unison_numpy(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']

#features: size, patch_size, patch_size, 1
#labels: size, patch_size, patch_size, 2. First is lesion, second is non-lesion
patch_features = np.zeros((database_size, patch_size, patch_size, 1), dtype=np.float32)
patch_labels = np.zeros((database_size, output_size, output_size, 2), dtype=np.float32)

f = []
l = []
for i in range(len(features)):
	for z in range(len(features[i])):
		f.append(features[i][z])
		l.append(labels[i][z])

features, labels = shuffle_in_unison(f, l)

patchSpan = patch_size // 2
outputSpan = output_size // 2
numLesions = 0
random.seed(time.time())
for n in range(database_size):
	success = False
	while(not success):
		if n < train_size:
			i = random.randrange(len(features)*9//10)
		else:
			i = random.randrange(len(features)*9//10, len(features))
		x = random.randrange(len(features[i]))
		y = random.randrange(len(features[i][x]))
		try:
			patch_features[n, :, :, 0] = features[i][x-patchSpan:x+patchSpan+1, y-patchSpan:y+patchSpan+1]
			patch_labels[n, :, :, 0] = labels[i][x-outputSpan:x+outputSpan+1, y-outputSpan:y+outputSpan+1]
			if(random.random() < flipRate):
				patch_features[n, :, :, 0] = cp.flipud(patch_features[n, :, :, 0])
				patch_labels[n, :, :, 0] = cp.flipud(patch_labels[n, :, :, 0])
			if(random.random() < flipRate):
				patch_features[n, :, :, 0] = cp.fliplr(patch_features[n, :, :, 0])
				patch_labels[n, :, :, 0] = cp.fliplr(patch_labels[n, :, :, 0])
		except:
			continue
		patch_labels[n, :, :, 1] = abs(1 - patch_labels[n, :, :, 0])

		if patch_labels[n, outputSpan, outputSpan, 0] > 0.5:
			numLesions += 1
			success = True

		if n * datasetPercentLesion <= numLesions:
			success = True;

train_features, valid_features = np.split(patch_features, [train_size])
train_labels, valid_labels = np.split(patch_labels, [train_size])

train_features, train_labels = shuffle_in_unison_numpy(train_features, train_labels)
valid_features, valid_labels = shuffle_in_unison_numpy(valid_features, valid_labels)

train_images = features[:len(features)*9//10]
valid_images = features[len(features)*9//10:]
train_images_labels = labels[:len(features)*9//10]
valid_images_labels = labels[len(features)*9//10:]

pickle_file = 'lesionDatabase.pickle'

print("Number of samples with lesions: ", numLesions)
try:
  f = open(pickle_file, 'wb')
  save = {
    'train_features': train_features,
    'train_labels': train_labels,
    'valid_features': valid_features,
    'valid_labels': valid_labels,
    'train_images': train_images,
    'valid_images': valid_images,
    'train_images_labels': train_images_labels,
    'valid_images_labels': valid_images_labels
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
