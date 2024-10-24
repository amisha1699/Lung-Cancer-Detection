# Lung-Cancer-Detection

## Introduction

This project focusses on developing a deep convolutional neural network model to detect lung cancer from CT-scan images. CT-scan images are different from X-Ray images as they provide a more detailed and three-dimensional view of the body's internal structures, allowing for better visualization of soft tissues, organs, and abnormalities. While X-ray images primarily capture two-dimensional projections, CT scans utilize multiple X-ray images taken from various angles and reconstruct them into cross-sectional images. This enhanced imaging capability is crucial in identifying small tumors, lesions, or nodules that might be overlooked in standard X-rays.

## Dataset

The source of the dataset is the LUNA16 dataset . The LUNA16 dataset is a subset of LIDC-IDRI dataset, in which the heterogeneous scans are filtered by different criteria.

## Model

The proposed model is a deep convolutional neural network. The architecture consists of several convolutional, pooling, and fully connected layers with dropout and batch normalization to improve generalization and reduce overfitting. Below is a detailed breakdown of the model architecture:

| Layer                   | Output Shape        | Parameters | Description                                               |
| ----------------------- | ------------------- | ---------- | --------------------------------------------------------- |
| **Input Layer**         | (None, 50, 50, 1)   | 0          | Input images of shape 50x50 with 1 channel (grayscale).   |
| **Conv2D**              | (None, 50, 50, 32)  | 320        | 32 filters, kernel size (3, 3), ReLU activation.          |
| **Batch Normalization** | (None, 50, 50, 32)  | 128        | Normalizes the activations from the Conv2D layer.         |
| **Conv2D**              | (None, 50, 50, 32)  | 9,248      | 32 filters, kernel size (3, 3), ReLU activation.          |
| **Batch Normalization** | (None, 50, 50, 32)  | 128        | Normalizes the activations from the Conv2D layer.         |
| **MaxPooling2D**        | (None, 25, 25, 32)  | 0          | 2x2 max pooling to reduce spatial dimensions.             |
| **Dropout**             | (None, 25, 25, 32)  | 0          | Dropout rate of 0.25 to reduce overfitting.               |
| **Conv2D**              | (None, 25, 25, 64)  | 18,496     | 64 filters, kernel size (3, 3), ReLU activation.          |
| **Batch Normalization** | (None, 25, 25, 64)  | 256        | Normalizes the activations from the Conv2D layer.         |
| **Conv2D**              | (None, 25, 25, 64)  | 36,928     | 64 filters, kernel size (3, 3), ReLU activation.          |
| **Batch Normalization** | (None, 25, 25, 64)  | 256        | Normalizes the activations from the Conv2D layer.         |
| **MaxPooling2D**        | (None, 12, 12, 64)  | 0          | 2x2 max pooling to reduce spatial dimensions.             |
| **Dropout**             | (None, 12, 12, 64)  | 0          | Dropout rate of 0.25 to reduce overfitting.               |
| **Conv2D**              | (None, 12, 12, 128) | 73,856     | 128 filters, kernel size (3, 3), ReLU activation.         |
| **Batch Normalization** | (None, 12, 12, 128) | 512        | Normalizes the activations from the Conv2D layer.         |
| **Conv2D**              | (None, 12, 12, 128) | 147,584    | 128 filters, kernel size (3, 3), ReLU activation.         |
| **Batch Normalization** | (None, 12, 12, 128) | 512        | Normalizes the activations from the Conv2D layer.         |
| **MaxPooling2D**        | (None, 6, 6, 128)   | 0          | 2x2 max pooling to reduce spatial dimensions.             |
| **Dropout**             | (None, 6, 6, 128)   | 0          | Dropout rate of 0.25 to reduce overfitting.               |
| **Conv2D**              | (None, 6, 6, 128)   | 147,584    | 128 filters, kernel size (3, 3), ReLU activation.         |
| **Batch Normalization** | (None, 6, 6, 128)   | 512        | Normalizes the activations from the Conv2D layer.         |
| **Conv2D**              | (None, 6, 6, 128)   | 147,584    | 128 filters, kernel size (3, 3), ReLU activation.         |
| **Batch Normalization** | (None, 6, 6, 128)   | 512        | Normalizes the activations from the Conv2D layer.         |
| **MaxPooling2D**        | (None, 3, 3, 128)   | 0          | 2x2 max pooling to reduce spatial dimensions.             |
| **Dropout**             | (None, 3, 3, 128)   | 0          | Dropout rate of 0.25 to reduce overfitting.               |
| **Conv2D**              | (None, 3, 3, 128)   | 147,584    | 128 filters, kernel size (3, 3), ReLU activation.         |
| **Batch Normalization** | (None, 3, 3, 128)   | 512        | Normalizes the activations from the Conv2D layer.         |
| **Conv2D**              | (None, 3, 3, 128)   | 147,584    | 128 filters, kernel size (3, 3), ReLU activation.         |
| **Batch Normalization** | (None, 3, 3, 128)   | 512        | Normalizes the activations from the Conv2D layer.         |
| **MaxPooling2D**        | (None, 1, 1, 128)   | 0          | 2x2 max pooling to reduce spatial dimensions.             |
| **Dropout**             | (None, 1, 1, 128)   | 0          | Dropout rate of 0.25 to reduce overfitting.               |
| **Flatten**             | (None, 128)         | 0          | Flattens the output for fully connected layers.           |
| **Dense**               | (None, 128)         | 16,512     | Fully connected layer with 128 units and ReLU activation. |
| **Batch Normalization** | (None, 128)         | 512        | Normalizes the activations from the dense layer.          |
| **Dropout**             | (None, 128)         | 0          | Dropout rate of 0.25 to reduce overfitting.               |
| **Dense**               | (None, 128)         | 16,512     | Fully connected layer with 128 units and ReLU activation. |
| **Dense**               | (None, 128)         | 16,512     | Fully connected layer with 128 units and ReLU activation. |
| **Dense**               | (None, 128)         | 16,512     | Fully connected layer with 128 units and ReLU activation. |
| **Dense (Output)**      | (None, 2)           | 258        | Output layer with 2 units (binary classification).        |

## Result

The model performs well on classifying both classes, with a strong accuracy and high precision and recall for class 0. The precision for class 1 could be improved slightly, but the overall performance indicates effective classification.
![Confusion Matrix](/confusion_mat.png)
