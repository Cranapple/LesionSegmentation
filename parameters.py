
import numpy as np

patch_size = 25 								#Size of patches sampled and used for training
output_size = 25 - 16 							#Size of output for the model
batch_size = 256 								#Size of batches during training
depth1 = 30 									#First depth variable for the model
depth2 = 40 									#Second depth variable for the model
depth3 = 50 									#Third depth variable for the model
num_hidden = 150 								#Number of hidden neurons. Not used by current models.
train_size = 100000 							#Size of the training dataset
valid_size = 10000 								#Size of the validation dataset
database_size = train_size + valid_size
numPatients = 23 								#Number of patients. Used during dataset generation.
datasetPercentLesion = 0.5 						#Percent of patches that must be centered on a lesion during database generation.
dropoutRate = 0.95 								#Rate for dropout in the models
l2Rate = 0.00001 								#L2 coefficient in the models
flipRate = 0.10 								#Percent rate at which patches are flipped during database generation.

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

def imgAccuracy(predictions, labels):
	return (np.sum(predictions == labels) / (predictions.shape[0] * predictions.shape[1]))

#Dice Score
def imgDSC(predictions, labels):
	TP = np.sum(np.logical_and(predictions == 1, labels == 1))
	FP = np.sum(np.logical_and(predictions == 1, labels == 0))
	P = np.sum(labels == 1)
	return TP / (FP + P + epsilon)