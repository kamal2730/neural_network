# NumPy Neural Networks from Scratch

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository showcases the implementation of Neural Networks (NNs) for image classification, built purely from scratch using **NumPy**. No high-level deep learning frameworks like TensorFlow, PyTorch, or Keras are used for the core NN logic, providing a deep dive into the fundamental mechanics of neural networks.

The project includes three distinct examples:
1.  Handwritten digit recognition (MNIST).
2.  Simpsons character classification (grayscale).
3.  Simpsons character classification (RGB, with more advanced features like dropout and regularization).

## Key Features

*   **NumPy-centric:** All core neural network components (layers, activations, loss functions, optimizer) are implemented using only NumPy.
*   **Modular Design:** Code is structured into classes for layers, activations, loss, and optimizers, making it understandable and extensible.
*   **Fundamental Components Implemented:**
    *   Dense (Fully Connected) Layer
    *   ReLU Activation
    *   Softmax Activation
    *   Categorical Cross-Entropy Loss
    *   Adam Optimizer
    *   Dropout Layer
    *   L1 & L2 Regularization (for weights and biases)
    *   Batch Generation
*   **Multiple Examples:** Demonstrates the network on different datasets and image types.
*   **Clear Visualizations:** Uses Matplotlib to plot training and validation loss/accuracy curves.
*   **Educational Focus:** Designed to help understand the inner workings of neural networks.

## Notebooks Included

1.  `Neural_Network_NumPy_Handwritten_detection.ipynb`:
    *   **Dataset:** MNIST (784 features per image, 10 classes for digits 0-9).
    *   **Architecture:**
        *   Input (784) -> Dense (512, L2 reg) -> ReLU -> Dropout (0.1) -> Dense (10) -> Softmax
    *   **Optimizer:** Adam
    *   **Purpose:** Classic handwritten digit recognition.

2.  `Neural_Network_NumPy_Simpson.ipynb`:
    *   **Dataset:** Simpsons Characters (Grayscale, 28x28 images = 784 features, 10 classes).
    *   **Architecture:**
        *   Input (784) -> Dense (512, L2 reg) -> ReLU -> Dropout (0.1) -> Dense (10) -> Softmax
    *   **Optimizer:** Adam
    *   **Purpose:** Basic character classification on custom grayscale images.

3.  `Neural_Network_NumPy_Simpson_rgb.ipynb`:
    *   **Dataset:** Simpsons Characters (RGB, 28x28x3 images = 2352 features, 10 classes). Uses augmented training data.
    *   **Architecture:**
        *   Input (2352) -> Dense (2048, L2 reg) -> ReLU -> Dropout (0.1)
        *   Dense (2048) -> Dense (512) -> ReLU -> Dropout (0.1)
        *   Dense (512) -> Dense (10) -> Softmax
    *   **Optimizer:** Adam
    *   **Purpose:** More complex character classification on custom RGB images, demonstrating a deeper network with dropout and regularization.

## Core Concepts Implemented

The notebooks implement the following neural network building blocks from scratch:

*   **Layers:**
    *   `Layer_Dense`: Standard fully connected layer.
    *   `Layer_Dropout`: Dropout for regularization.
*   **Activations:**
    *   `Activation_ReLU`: Rectified Linear Unit.
    *   `Activation_Softmax`: For multi-class classification output.
*   **Loss Function:**
    *   `Loss_CategoricalCrossentropy`: For multi-class classification.
    *   `Activation_Softmax_Loss_CategoricalCrossentropy`: A combined layer for efficient backpropagation with Softmax and CCE.
*   **Optimizer:**
    *   `Optimizer_Adam`: Adaptive Moment Estimation optimizer.
*   **Regularization:**
    *   L1 and L2 regularization for weights and biases within the `Layer_Dense` and `Loss` classes.
*   **Data Handling:**
    *   Custom functions for loading and preprocessing image data (PIL, NumPy).
    *   Batch generator for training.

## Why NumPy?

Building these components with NumPy allows for a granular understanding of:
*   Forward and backward propagation.
*   Gradient calculations.
*   Weight and bias updates.
*   The role of activation functions and loss functions.
*   How optimizers like Adam adjust learning rates.
*   The impact of regularization techniques like Dropout and L1/L2.

## Prerequisites

*   Python 3.9+
*   NumPy
*   Matplotlib
*   Scikit-learn (for MNIST dataset fetching and `shuffle`)
*   Pillow (PIL) (for image loading in Simpsons notebooks)

You can install these dependencies using pip:
```bash
pip install numpy matplotlib scikit-learn Pillow jupyter