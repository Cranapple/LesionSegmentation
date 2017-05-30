
import numpy as np

patch_size = 25
output_size = 25 - 16 #Change depending on number of conv.
batch_size = 256
depth1 = 30
depth2 = 40
depth3 = 50
num_hidden = 150
kernel_size = 5
train_size = 100000
valid_size = 10000
database_size = train_size + valid_size
numPatients = 23
datasetPercentLesion = 0.5

epsilon = 1e-6

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) / (predictions.shape[0] * predictions.shape[1] * predictions.shape[2]))

#Dice Score
def DSC(predictions, labels):
	TP = np.sum(np.logical_and(np.argmax(predictions, 3) == 1, np.argmax(labels, 3) == 1))
	FP = np.sum(np.logical_and(np.argmax(predictions, 3) == 1, np.argmax(labels, 3) == 0))
	P = np.sum(np.argmax(labels, 3) == 1)
	return 100 * TP / (FP + P + epsilon)

def percentLesion(labels):
	shape = labels.shape
	sum = shape[0] * shape[1] * shape[2]
	return (sum - np.sum(np.argmax(labels, 3))) / sum
