# Kannada Digit Recognition using CNN and KNN

This repository contains a machine learning project focused on recognizing
handwritten Kannada digits (0-9) using two primary classifiers: Convolutional
Neural Networks (CNN) and K-Nearest Neighbors (KNN). The project was implemented
entirely in MATLAB, leveraging various preprocessing and augmentation techniques
to optimize model performance.

## Overview

The Kannada MNIST dataset is an adaptation of the classic MNIST digit
recognition task, consisting of 75,000 grayscale images (28x28 pixels) of
handwritten Kannada numerals. The objective was to accurately classify these
digits using supervised machine learning techniques.

### Dataset

* Total Images: ~75,000 (60,000 training, 10,000 validation, 5,000 testing)

* Image Dimensions: 28x28 pixels, Grayscale

## Preprocessing Techniques

### Normalization

* Rescaled pixel intensity values to the range [0, 1] to standardize image
  brightness and reduce noise.

### Principal Component Analysis (PCA)

* Reduced dimensionality from 784 pixels to 237 principal components, explaining
  95% variance.

* Intended for use with KNN to simplify computation and remove less informative
  features.

### Canny Edge Detection

* Identified image edges to emphasize the shape and outline of digits.

* Reduced complexity of image data for KNN classification.

### Data Augmentation

Increased dataset from 60,000 to 120,000 images through augmentation techniques:

* Random rotation [-45°, 45°]

* Random scaling [0.75, 1.25]

* Random translations [-2, 2] pixels

Improved robustness and generalization of CNN model.

## Machine Learning Models

### K-Nearest Neighbors (KNN)

* Evaluated with varying values of k (3, 5, 7).

* Tested preprocessing combinations including PCA, Canny edge detection, and
  their combination.

* Highest accuracy obtained with raw (normalized/non-normalized) dataset
  (~71.9%).

### Convolutional Neural Network (CNN)

Consisted of:

* 1 Input Layer (28x28)

* 3 Convolutional Layers (3x3 kernel, Batch Normalization, ReLU activation, Max
	Pooling)

* 1 Fully Connected Layer (10 output neurons, Softmax activation)

* Achieved 75.57% accuracy with normalized, non-augmented dataset.

* Achieved 87.28% accuracy with augmented dataset.

## Ablation Study

Conducted experiments by varying the number of convolutional layers in the CNN:

| Convolutional Layers | Augmented Dataset Accuracy (%) | Non-Augmented Dataset Accuracy (%) |
|----------------------|--------------------------------|------------------------------------|
| Four Layers          | 85.31                          | 76.77                              |
| Three Layers         | 87.28                          | 75.57                              |
| Two Layers           | 84.22                          | 72.90                              |
| One Layer            | 77.54                          | 70.01                              |

Three-layer CNN architecture showed the best performance.

## Future Directions

* Explore alternative preprocessing techniques like Scale-Invariant Feature
  Transform (SIFT) for improving KNN performance.

* Enhance dataset augmentation strategies to further improve CNN robustness and
  accuracy.

* Conduct training using GPU-accelerated computing environments for faster
  experimentation and optimization.

Tools Used • MATLAB • MATLAB Deep Learning Toolbox

Author • Shawn Kim tkim1@arizona.edu

⸻

Note: For detailed insights, visualization, and methodology, refer to the
included [project report](final_project.pdf).
