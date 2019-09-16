# Dynamic-Hand-Gesture-Recognition-using-CNN
CNN based hand recognizer
The model is trained on 4 gestures: two of them are static and two of them are dynamic. Process for dynamic gestures:
a.Give and input video od the gesture to extractingFrames.py
b. give the results to preprocessing.py
c. after that the images are fed to CNN model

**extractFrames.py**
This file extracts frames from video input.

**preprocessing.py**
This file does the following pre-processing to speed up CNN:
a. applies median blur
b. applies adaptive gaussian thresholding
c. changes image size to 128x128

Results of preprocessing the images
![Preprocessing result](https://github.com/kreeteeekah/Dynamic-Hand-Gesture-Recognition-using-CNN/blob/master/original.JPG)

Results of training the model
![Snapshot of model accuracy](https://github.com/kreeteeekah/Dynamic-Hand-Gesture-Recognition-using-CNN/blob/master/MainTest1JPG.JPG)




