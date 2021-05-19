---
title: "Hand Gesture Recognition for Mouse-Free Assistive Technology"
# image_url: /assets/ROAR-bg.png
keywords: [Machine Learning, Computer Vision]
---

## Summary

For my master's capstone project at UC Berkeley, I worked in [Professor Brian A. Barsky](https://people.eecs.berkeley.edu/~barsky/)'s [Mouse-Free Assistive Technology](https://barskygroup.wixsite.com/home/research-areas) team. Our final report can be accessed [here](/assets/capstone_report.pdf). This article summarizes my work in this group project.

The goal of our project was to build a system to control a computer cursor using the camera input. We aimed to make the system portable enough to run on an affordable CPU. I was responsible for hand gesture classification, which is an active research area in computer vision. I applied several approaches, including a 3D-CNN model and several keypoint-based models.

## 3D-CNN-Based Hand Gesture Recognition

Recognizing the gestures represented by a sequence of image frames requires large amounts of computations. Thanks to the advance of deep convolutional neural networks (CNNs) and recurrent neural networks (RNNs) such as Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU), new techniques based on large-sized CNN models tend to achieve high accuracy in gesture classification tasks. Most approaches use either 3D-CNNs or RNNs (LSTMs) to extract temporal information.

3D-CNN-based models perform convolutional operations along the time axis, which enables them to learn rich temporal information. [Kopuklu et al.](https://arxiv.org/abs/1901.10323) achieved state-of-the-art performance using a lightweight CNN model to detect gestures and a deep 3D-CNN model to classify the detected gestures. The ResNeXt-101 model, which is used as a classifier, achieved the state-of-the-art offline classification accuracy of 94.04% on the EgoGesture dataset. However, extracting human pose information from images requires a large amount of computation. I quantized this model using [PyTorch's Quantization module](https://pytorch.org/docs/stable/quantization.html) and ran it on my MacBook 2019 with an 8-Core Intel Core i9. The inference time was above 0.3 seconds, which is not acceptable for cursor control.

## Keypoint-Based Hand Gesture Recognition

Coordinates of body keypoints are a more compact and meaningful representation. Skeleton-based hand gesture recognition approaches use coordinates of pre-defined hand keypoints as input and output labels of hand gestures. Skeleton-based hand gesture recognition has two advantages over image-based approaches: First, the location of hand could be computed from keypoints that could be used for hand tracking, and thus I did not need a separate palm detector. Second, the skeleton data has much fewer dimensions than image data, and its information can be extracted by shallow neural networks with lower computational cost compared to CNN on images.

<!-- These approaches have much smaller model sizes and require much less computation compared to image-based approaches. [De Smedt et al.'s approach](https://hal.archives-ouvertes.fr/hal-01535152) used a temporal pyramid to obtain the feature vectors from a multi-level representation of Fisher Vectors and other skeleton-based geometric features. It then completed the classification task using a linear SVM classifier. [Nunez et al.](https://www.sciencedirect.com/science/article/pii/S0031320317304405) proposed a model that combines a Convolutional Neural Network (CNN) and a Long-Short Term Memory (LSTM) recurrent network for handling time series of 3D coordinates of skeleton keypoints. [Double-feature Double-motion Network (DD-Net)](http://arxiv.org/abs/1907.09658) uses a Joint Collection Distances (JCD) feature and a two-scale global motion feature. -->

The procedure of skeleton-based hand gesture recognition is as follows: First, I used a hand keypoint detector to extract the skeleton of the user's hand from a video stream. Then, I pre-processed the coordinates of keypoints and generate features. Finally, I used the feature as the input to a classifier which maps skeleton data to the probability of each output label, i.e., gesture.

My hand keypoint detector had the same structure as MediaPipe Hands, a real-time on-device hand tracking tool. It included a palm detector and a hand landmark model. The former operates on a full input image and located palms via an oriented hand bounding box, while the latter operated on the cropped hand bounding box provided by the palm detector and returns 21 landmarks consisting of x, y, and relative depth.

Although neural networks complex enough are able to learn latent representations in the coordinates for gesture classification tasks, past research finds that handmade features are useful for gesture classification. In my experiments, I found that introducing certain features could effectively improve the classification accuracy without increasing the inference time. For each video frame, I extracted feature vectors consisting of angles between the connections of pairs of joints and the length of the connections.

In order to keep the inference time short on CPUs, I used a simple RNN-based neural network to classify hand gestures using these features. I designed and tested two approaches to use the feature sequences: sliding window and stateful RNN. In the sliding window approach, I simply used the features in the last k frames as the input and reset the hidden states for every frame. In the stateful RNN approach, I kept the hidden states and only fed the features in the last frame into the network. The sliding window approach had better accuracy in general, but its cost is higher by _sequence length_ times. I found that the model consisting of a single GRU cell and a softmax layer gave decent and robust classification accuracy in both settings.

To train such a classifier, I used a subset of EgoGesture with 11 static gestures: fist and ten gestures representing digits from 0 to 9. I chose the sequence length to be 5 and the frame rate of input to be 10 FPS. Recurrent neural network using a single layer of GRU reached the highest accuracy of 83.25% with the sequence to sequence models and 77.66% with the stateful RNN models. Although LSTM and GRU had similar accuracy in multiple experiments, GRU had a simpler structure and gave much smaller latency. Most false predictions happen when our keypoint detector cannot detect keypoints precisely, especially when doing gestures like seven and nine (see below).

<img src="/assets/img/seven.jpg" width="49%" >
<img src="/assets/img/nine.jpg" width="49%">

## Conclusion and Future work

Although keypoint-based hand gesture classifier's accuracy was lower than the state-of-the-art 3D-CNN's, the model was much simpler and enables the whole program to run on CPU with an acceptable latency. The model I used to extract keypoints did not work well for certain hand gestures. Future works could consider using other methods to extract hand keypoints in an image.
