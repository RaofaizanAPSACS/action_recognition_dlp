# Violence Detection in Surveillance Videos

## Objective
The objective of this project is to develop a practical system for automatically detecting violence in surveillance videos using deep learning techniques. The increasing deployment of surveillance cameras necessitates the development of efficient automated systems for real-time violence detection and identification. These systems would significantly enhance the ability of law enforcement to intervene and apprehend perpetrators.

## Problem Statement
The increase in public violence, particularly in settings like high schools and streets, necessitates the deployment of surveillance systems. However, existing systems often require human inspection of video footage to identify violent incidents, which is inefficient. This project aims to address this inefficiency by developing a system capable of automatically monitoring and identifying violent events in surveillance videos without human intervention.

## Methodology

### Data Collection and Preprocessing
- The dataset consists of surveillance videos containing instances of violence and non-violence. Frames from these videos are extracted and resized to a suitable format for input to the neural network.

### Pre-Trained Models
- **VGG16**: The VGG16 model, pre-trained on the ImageNet dataset, is used to extract high-level features from video frames. The output of the pre-trained VGG16 model is fed into a dense layer to obtain transfer values.
- **MobileNet**: The MobileNet model, pre-trained on the ImageNet dataset, is used to extract high-level features from video frames. The output of the pre-trained MobileNet model is fed into a dense layer to obtain transfer values.

### Recurrent Neural Network (LSTM)
- The transfer values from VGG16 and MobileNet are input to a Long Short-Term Memory (LSTM) network. LSTM is chosen for its ability to analyze temporal information in the video frames. The LSTM architecture is designed with appropriate input dimensions to process the 20 frames per video.

### Model Training
- The LSTM network is trained using a subset of the dataset, with epochs and batch size specified. The model is trained to classify videos as violent or non-violent based on the features extracted by VGG16 and processed by LSTM.

### Model Evaluation
- The trained model is evaluated on a separate test set comprising 20% of the total videos. Performance metrics such as loss and accuracy are computed to assess the model's effectiveness in violence detection.

## Results
The trained model achieved a validation accuracy of 93.5% on the test set, demonstrating its efficacy in automatically detecting violence in surveillance videos. The model's loss and accuracy trends over epochs are visualized to provide insights into its training dynamics.

## References
- Simonyan, K., & Zisserman, A. (2015). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Chollet, F. et al. Keras.
- OpenCV Library Documentation.
- Matplotlib Documentation.
