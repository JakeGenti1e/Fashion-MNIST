# Fashion-MNIST CNN Classification

## Overview

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify images from the Fashion-MNIST dataset into 10 clothing categories. 

The goal of this project was to add onto my previous CNN experience by working with a more challenging image-classification problem and analyzing the successes and failures of the model.

## Dataset

Fashion-MNIST contains 70,000 grayscale 28x28 images accross 10 classes:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

The dataset is divided into 60,000 training images and 10,000 test images.

## Model Architecture

The model uses a convolutional neural network that's comprised of: 

- Convolutional layers for feature extraction
- ReLU activation functions
- Max-pooling layers for spatial downsampling
- Fully connected layers for classification
- 10 output neurons that correspond to the 10 Fashion-MNIST classes

## Training

Loss Function: CrossEntropyLoss
Optimizer: Adam
Learning Rate: 0.001
Batch Size: 64
Epochs: 15

## Results
Final Test Accuracy: 90.35%

The model was able to successfully learn to identify most Fashion-MNIST categories. Performance differed accross classes, especially for clothing categories that were visually similar to each other.

## Confusion Matrix


  
