# Violence Detection using CNN + LSTM Neural Network

## Introduction
The prevalence of public violence has surged in recent times, necessitating the widespread use of surveillance cameras for monitoring. However, manually inspecting surveillance videos to identify violent incidents is inefficient. This project aims to automate the process using deep learning techniques.

## Flowchart
The method involves extracting frames from videos, passing them through a pre-trained VGG16 network, and then using the output of a final layer as input to an LSTM network for temporal analysis and violence detection.


## Dependencies
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- h5py

## Model Architecture
- VGG16 CNN for feature extraction
 ![image](https://github.com/MuhammadUmerHussain/HospitalWeb/assets/108338561/809bac69-bb8b-4d41-906c-5a6d17ebd100)

- LSTM for temporal analysis
  ![image](https://github.com/MuhammadUmerHussain/HospitalWeb/assets/108338561/506274b8-5657-4225-a6df-d20a264cfc5a)

- Dense layers for classification


## Installation

python
pip install tensorflow keras opencv-python



## Imports
python
%matplotlib inline
import cv2
import os
import numpy as np
import keras
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Activation
import sys
import h5py

## Results
- Achieved accuracy: 93.5% on test data
- Model loss and accuracy visualizations provided in results/ directory

## References
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf)
