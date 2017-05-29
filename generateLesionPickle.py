import dicom
import os
import numpy
from matplotlib import pyplot
from six.moves import cPickle as pickle
from parameters import *

thresholdVal = 5000
featureArray = []
labelArray = []
for i in range(1, numPatients + 1):
	PathFeatures = "./LesionDataset/" + str(i) + "/Features"
	PathLabels = "./LesionDataset/" + str(i) + "/Labels"

	lstFilesDCM = []
	for dirName, subdirList, fileList in os.walk(PathFeatures):
		for filename in fileList:
			if ".dcm" in filename.lower():
				lstFilesDCM.append(os.path.join(dirName,filename))
	RefDs = dicom.read_file(lstFilesDCM[0])
	ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))
	ArrayDicom = numpy.zeros(ConstPixelDims, dtype=numpy.float32)
	for filenameDCM in lstFilesDCM:
		ds = dicom.read_file(filenameDCM)
		img = ds.pixel_array
		img = img - numpy.mean(img)		#normalize
		img = img / numpy.std(img)
		ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = img #ds.pixel_array 
	featureArray.append(ArrayDicom)

	lstFilesDCM = []
	for dirName, subdirList, fileList in os.walk(PathLabels):
		for filename in fileList:
			if ".dcm" in filename.lower():
				lstFilesDCM.append(os.path.join(dirName,filename))
	RefDs = dicom.read_file(lstFilesDCM[0])
	ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))
	ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
	for filenameDCM in lstFilesDCM:
		ds = dicom.read_file(filenameDCM)
		ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array
	lowVal = ArrayDicom < thresholdVal
	highVal = ArrayDicom >= thresholdVal
	ArrayDicom[lowVal] = 0
	ArrayDicom[highVal] = 1
	labelArray.append(ArrayDicom)

pickle_file = 'lesion.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'features': featureArray,
    'labels': labelArray
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
