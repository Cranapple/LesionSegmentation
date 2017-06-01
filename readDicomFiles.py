import dicom
import os
import numpy
from matplotlib import pyplot
#import cv2

PathDicom = "./LesionDataset/1/Labels"
imgNum = 0;
lstFilesDCM = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():
            lstFilesDCM.append(os.path.join(dirName,filename))

RefDs = dicom.read_file(lstFilesDCM[0])

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
print(ConstPixelDims)

ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

for filenameDCM in lstFilesDCM:
    ds = dicom.read_file(filenameDCM)
    print(filenameDCM)
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array 

img = ArrayDicom[:, :, imgNum]
pyplot.imshow(img, cmap='gray')
pyplot.show()