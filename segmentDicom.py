import os
import numpy as np
from matplotlib import pyplot
from six.moves import cPickle as pickle
from parameters import *
from numpy import random
import tensorflow as tf
import math
import dicom
import sys

PathDicom = "./LesionDataset/5/Features/IM-0001-0017-0001.dcm"
modelName = "833CNNnorm"
step = 6000

threshold = 25

def mean(img):
	img = img.flatten()
	sum, px = 0, 0
	for p in img:
		if p >= threshold:
			sum += p
			px += 1
	return sum / px

def stddev(img):
	m = mean(img)
	img = img.flatten()
	sum, px = 0, 0
	for p in img:
		if p >= threshold:
			sum += (p - m)*(p - m)
			px += 1
	return np.sqrt(sum / px)


RefDs = dicom.read_file(PathDicom)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
imgF = np.zeros(ConstPixelDims, dtype=np.float32)
ds = dicom.read_file(PathDicom)
imgF[:, :] = ds.pixel_array

imgF = imgF - mean(imgF)		#normalize
imgF = imgF / stddev(imgF)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:
	new_saver = tf.train.import_meta_graph('.\lesion_models\\' + modelName + "\\" + modelName + "-" + str(step) + ".meta")
	new_saver.restore(session, tf.train.latest_checkpoint('.\lesion_models\\' + modelName))
	graph = tf.get_default_graph()
	tf_test_features = graph.get_tensor_by_name("features:0")
	test_prediction = graph.get_tensor_by_name("labels:0")

	imgPL = np.zeros(imgF.shape, dtype=np.float32)
	imgPL2 = np.zeros(imgF.shape, dtype=np.float32)
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
				imgPL2[xP:imgPL.shape[0], yP:imgPL.shape[1]]  = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, 0:imgPL.shape[1]-yP, 0] > 0.5
			elif x == xDim - 1:
				imgPL[xP:imgPL.shape[0], yP:yP+output_size] = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, :, 0]
				imgPL2[xP:imgPL.shape[0], yP:yP+output_size] = outputs[x*yDim + y, 0:imgPL.shape[0]-xP, :, 0] > 0.5
			elif y == yDim - 1:
				imgPL[xP:xP+output_size, yP:imgPL.shape[1]] = outputs[x*yDim + y, :, 0:imgPL.shape[1]-yP, 0]
				imgPL2[xP:xP+output_size, yP:imgPL.shape[1]] = outputs[x*yDim + y, :, 0:imgPL.shape[1]-yP, 0] > 0.5
			else:
				imgPL[xP:xP+output_size, yP:yP+output_size] = outputs[x*yDim + y, :, :, 0]
				imgPL2[xP:xP+output_size, yP:yP+output_size] = outputs[x*yDim + y, :, :, 0] > 0.5

	pyplot.subplot(131)
	pyplot.imshow(imgF, cmap='gray')
	pyplot.subplot(132)
	pyplot.imshow(imgPL, cmap='gray')
	pyplot.subplot(133)
	pyplot.imshow(imgPL2, cmap='gray')
	pyplot.show()