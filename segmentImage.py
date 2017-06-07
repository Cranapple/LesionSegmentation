import os
import numpy as np
from matplotlib import pyplot
from six.moves import cPickle as pickle
from parameters import *
from numpy import random
import tensorflow as tf
import math
from skimage.morphology import disk, opening
import time

#---------------------------------------------------------------------------------

#Program to run and vizualize a segmentation on a image using the saved model

numPatchSamples = 0				#Patches
numImgSamples = 0				#Images
numTestSamplePatches = 0		#Prediction Patches
numTestSamples = 0				#Prediction Images
modelName = "833CNN"
step = 10000
heatMap = True
useValid = True;
accuracyScore = True;
usePostProc = True;

#---------------------------------------------------------------------------------

selem = disk(4)

#Testing the dataset

pickle_file = 'lesionDatabase.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_features = save['train_features']
	train_labels = save['train_labels']
	valid_features = save['valid_features']
	valid_labels = save['valid_labels']
	valid_images = save['valid_images']
	train_images = save['train_images']
	valid_images_labels = save['valid_images_labels']
	train_images_labels = save['train_images_labels']

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']

if useValid:
	s = valid_size
else:
	s = train_size

#TEST PATCHES
for i in random.choice(s, numPatchSamples):
	if useValid:
		img1 = valid_features[i, :, :, 0]
		img2 = valid_labels[i, :, :, 0]
		img3 = valid_labels[i, :, :, 1]
	else:
		img1 = train_features[i, :, :, 0]
		img2 = train_labels[i, :, :, 0]
		img3 = train_labels[i, :, :, 1]
	img = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
	img[:, :, 0] = img1
	img[8:17, 8:17, 1] = img2
	img[8:17, 8:17, 2] = img3
	pyplot.imshow(img)
	pyplot.show()

#TEST IMAGES
for i in random.choice(numPatients, numImgSamples):
	for z in random.choice(len(features[i]), 1):
		img = features[i][z]
		pyplot.subplot(121)
		pyplot.imshow(img, cmap='gray')
		img = labels[i][z]
		pyplot.subplot(122)
		pyplot.imshow(img, cmap='gray')
		pyplot.show()


#Testing the model

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
	new_saver = tf.train.import_meta_graph('.\lesion_models\\' + modelName + "\\" + modelName + "-" + str(step) + ".meta")
	new_saver.restore(session, tf.train.latest_checkpoint('.\lesion_models\\' + modelName))
	graph = tf.get_default_graph()
	tf_test_features = graph.get_tensor_by_name("features:0")
	test_prediction = graph.get_tensor_by_name("labels:0")

	#TEST PATCH PREDICTIONS
	for i in random.choice(s, numTestSamplePatches):
		if useValid:
			img1 = valid_features[i, :, :, 0]
			img2 = valid_labels[i, :, :, 0]
			img3 = valid_labels[i, :, :, 1]
		else:
			img1 = train_features[i, :, :, 0]
			img2 = train_labels[i, :, :, 0]
			img3 = train_labels[i, :, :, 1]
		img = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
		img[:, :, 0] = img1
		img[8:17, 8:17, 1] = img2
		img[8:17, 8:17, 2] = img3
		pyplot.subplot(121)
		pyplot.imshow(img)

		f = img1.reshape((1, patch_size, patch_size, 1))
		feed_dict = {tf_test_features : f}
		output = session.run([test_prediction], feed_dict=feed_dict)
		output = np.array(output).reshape((1, output_size, output_size, 2))
		if heatMap:
			img2 = output[0, :, :, 0]
		else:
			img2 = output[0, :, :, 0] > 0.5
		img3 = 1 - img2
		img = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
		img[:, :, 0] = img1
		img[8:17, 8:17, 1] = img2
		img[8:17, 8:17, 2] = img3
		pyplot.subplot(122)
		pyplot.imshow(img)

		pyplot.show()

	#TEST FULL IMAGE PREDICTIONS
	if useValid == True:
		s = len(valid_images)
	else:
		s = len(train_images)
	for i in random.choice(s, numTestSamples):
		if useValid == True:
			imgF = valid_images[i]
			imgL = valid_images_labels[i]
		else:
			imgF = train_images[i]
			imgL = train_images_labels[i]
		imgPL = np.zeros(imgL.shape, dtype=np.float32)
		imgPL2 = np.zeros(imgL.shape, dtype=np.float32)
		xDim = math.ceil(imgF.shape[0]  / output_size)
		yDim = math.ceil(imgF.shape[1]  / output_size)
		padSize = (patch_size-output_size)//2
		padX = xDim*output_size+padSize - imgF.shape[0]
		padY = yDim*output_size+padSize - imgF.shape[1]
		imgF2 = np.pad(imgF, ((padSize,  padX), (padSize, padY)), "minimum")
		patches = np.zeros((xDim*yDim, patch_size, patch_size, 1), dtype=np.float32)
		for x in range(xDim):
			for y in range(yDim):
				patches[x * yDim + y, :, :, 0] = imgF2[x*output_size:x*output_size+patch_size, y*output_size:y*output_size+patch_size]

		feed_dict = {tf_test_features : patches}
		outputs = session.run([test_prediction], feed_dict=feed_dict)
		outputs = np.array(outputs).reshape((-1, output_size, output_size, 2))
		for x in range(xDim):
			for y in range(yDim):
				xP = x*output_size
				yP = y*output_size
				if x == xDim - 1 and y == yDim - 1:
					imgPL[xP:imgPL.shape[0], yP:imgPL.shape[1]] = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, 0:imgPL.shape[1]-yP, 0]
				elif x == xDim - 1:
					imgPL[xP:imgPL.shape[0], yP:yP+output_size] = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, :, 0]
				elif y == yDim - 1:
					imgPL[xP:xP+output_size, yP:imgPL.shape[1]] = outputs[x*yDim + y, :, 0:imgPL.shape[1]-yP, 0]
				else:
					imgPL[xP:xP+output_size, yP:yP+output_size] = outputs[x*yDim + y, :, :, 0]
		if usePostProc:
			imgPL = opening(imgPL, selem)
		imgPL2 = imgPL > 0.5
		pyplot.subplot(141)
		pyplot.imshow(imgF, cmap='gray')
		pyplot.subplot(142)
		pyplot.imshow(imgL, cmap='gray')
		pyplot.subplot(143)
		pyplot.imshow(imgPL, cmap='gray')
		pyplot.subplot(144)
		pyplot.imshow(imgPL2, cmap='gray')
		pyplot.show()

	#ACCURACY SCORE
	if accuracyScore:
		sumAcc = 0;
		sumDSC = 0;
		n = 0;
		start_time = time.time()
		for i in range(len(valid_images)):
			imgF = valid_images[i]
			imgL = valid_images_labels[i]
			if np.sum(imgL) < 0.5:
				continue;
			imgPL = np.zeros(imgL.shape, dtype=np.float32)
			#imgPL2 = np.zeros(imgL.shape, dtype=np.float32)
			xDim = math.ceil(imgF.shape[0]  / output_size)
			yDim = math.ceil(imgF.shape[1]  / output_size)
			padSize = (patch_size-output_size)//2
			padX = xDim*output_size+padSize - imgF.shape[0]
			padY = yDim*output_size+padSize - imgF.shape[1]
			imgF2 = np.pad(imgF, ((padSize,  padX), (padSize, padY)), "minimum")
			patches = np.zeros((xDim*yDim, patch_size, patch_size, 1), dtype=np.float32)
			for x in range(xDim):
				for y in range(yDim):
					patches[x * yDim + y, :, :, 0] = imgF2[x*output_size:x*output_size+patch_size, y*output_size:y*output_size+patch_size]

			feed_dict = {tf_test_features : patches}
			outputs = session.run([test_prediction], feed_dict=feed_dict)
			outputs = np.array(outputs).reshape((-1, output_size, output_size, 2))
			for x in range(xDim):
				for y in range(yDim):
					xP = x*output_size
					yP = y*output_size
					if x == xDim - 1 and y == yDim - 1:
						imgPL[xP:imgPL.shape[0], yP:imgPL.shape[1]] = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, 0:imgPL.shape[1]-yP, 0]
					elif x == xDim - 1:
						imgPL[xP:imgPL.shape[0], yP:yP+output_size] = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, :, 0]
					elif y == yDim - 1:
						imgPL[xP:xP+output_size, yP:imgPL.shape[1]] = outputs[x*yDim + y, :, 0:imgPL.shape[1]-yP, 0]
					else:
						imgPL[xP:xP+output_size, yP:yP+output_size] = outputs[x*yDim + y, :, :, 0] 
			if usePostProc:
				imgPL = opening(imgPL, selem)
			imgPL = imgPL > 0.5
			sumAcc += imgAccuracy(imgPL, imgL)
			sumDSC += imgDSC(imgPL, imgL)
			n += 1
		print("Average Time per Image: ", (time.time() - start_time) / n)
		print("Average Accuracy: ", sumAcc / n)
		print("Average DSC: ", sumDSC / n)
		print("\n")