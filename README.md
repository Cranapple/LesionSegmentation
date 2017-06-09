# LesionSegmentation

The code for the MRI FLAIR lesion segmentation project. Constructed for UCLA's CS 188 Medical Imaging Class.
The report is submitted on CCLE and can also be found [here](https://github.com/Cranapple/LesionSegmentation/blob/master/report.pdf).
Our dataset, along with our version of the generated files, will be shared directly with the instructor.
I give permission for this work to be shared to reviewers for award consideration.

### Files
**generateLesionPickle.py** - Generates the pickle file containing brain scans.

**generateDatabasePickle.py** - Generates the pickle file containing the image patches used for training. Relies on the previous script's generated file.

**parameters.py** - Constants used throughout each script.

**segmentDicom.py** - File used to run the model on a standalone image. Takes in a dicom image filepath as the first argument and displays images of the predicted segmentation.

**segmentImage.py** - Script used to observe data examples, run test predicitons, and calculate overall evaluation measures.

**train\<model>.py** - Files used to train models and generate tensorboard outputs.

### Using New Data
In order to train a model with your own data, you will need to format your data as follows. In a folder titled "LesionDataset", input folders named "1", "2", "3", etc. up to your total number of patients. Then in each folder, place dicom images in directories named "Features" and "Labels". The Features folder must contain the original dicom images, and the labels folder must contain dicom images in which values above the threshold are equivalent to a lesion pixel and ones below are non-lesions. The threshold is set to 5000, but this can be changed by a constant in the generateLesionPickle.py script. The images must be ordered sequentially such that pairs from each folder are processed together.
Throughout the entire process, elements of the model and data can be altered in the parameter.py file.

### Database Generation
Once you have followed the instructions in the previous section, run generateLesionPickle.py, followed by running generateDatabasePickle.py. This should create files named lesion.pickle and lesionDatabase.pickle in your directory.

### Training
Run one of the train files to train the respective model. This will also generate tensorboard data that can accessed during and after training. To do such, run the following command and enter "http://localhost:6006/" into your browser. Take care to replace the model name with the one you are using.
```
tensorboard --logdir=./lesion_tensorboard/<ModelName>
```
Tensorboard data should be deleted between runs, unless you wish to have data from two seperate runs go into the same graphs.
### Using the Models
There are two ways to use the models for predictions. The first is segementDicom.py. This script takes in a filepath to the dicom image you want to segment as its first and only parameter, and will display the result as produced by the model named in the modelName variable.
The second way is with segmentImage.py. This script is meant to be used to diagnose, visualize, and evaluate your model. It contains constants at the top of the file which are meant to be changed by the user for his or her needs. This script can display images and patches from your dataset, output predictions for images and patches in your dataset, and produce average accuracy, similarity and time spent for your model.

### Dependencies
[numpy](http://www.numpy.org/): Used for tensorflow and other libraries.

[matplotlib](https://matplotlib.org/): For pyplot.

[six](https://pythonhosted.org/six/): For pickle.

[tensorflow](https://www.tensorflow.org/): Library used for deep learning.

[skimage](http://scikit-image.org/): Used for image opening post-processing.

[dicom](http://pydicom.readthedocs.io/en/stable/getting_started.html): Used to open dicom files.


Developed on a Windows 10 machine with Python 3.5.2
