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
![image](https://github.com/RaofaizanAPSACS/action_recognition_dlp/assets/51090356/5f0c6321-0e67-44dd-be90-f7ed51dac977)


- LSTM for temporal analysis
 ![image](https://github.com/RaofaizanAPSACS/action_recognition_dlp/assets/51090356/c9ec7efb-1615-4742-b066-225159ed302a)


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
