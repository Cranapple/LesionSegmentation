import dicom
import os
import numpy
from matplotlib import pyplot

thresholdVal = 5000
featureArray = []
labelArray = []
for i in range(1, 13):
	PathFeatures = "./LesionDataset/" + str(i) + "/Features"
	PathLabels = "./LesionDataset/" + str(i) + "/Labels"

	lstFilesDCM = []
	for dirName, subdirList, fileList in os.walk(PathFeatures):
		for filename in fileList:
			if ".dcm" in filename.lower():
				lstFilesDCM.append(os.path.join(dirName,filename))
	RefDs = dicom.read_file(lstFilesDCM[0])
	ConstPixelDims = (len(lstFilesDCM), int(RefDs.Rows), int(RefDs.Columns))
	ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
	for filenameDCM in lstFilesDCM:
		ds = dicom.read_file(filenameDCM)
		ArrayDicom[lstFilesDCM.index(filenameDCM), :, :] = ds.pixel_array 
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

i = 9
k = 13
#for i in range(len(featureArray)):
#	for k in range(len(featureArray[i])):
img = featureArray[i][k]
pyplot.imshow(img, cmap='gray')
pyplot.draw()
pyplot.show()
img = labelArray[i][k]
pyplot.imshow(img, cmap='gray')
pyplot.draw()		
pyplot.show()
