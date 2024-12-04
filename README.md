# Brain Tumor Classification using Deep Learning

This project implements a convolutional neural network (CNN) to classify brain tumor images into two categories: `yes` (tumor) and `no` (no tumor). The model is trained on a dataset of MRI images and achieves binary classification through grayscale image processing. 

---

## Table of Contents

1. [Overview](#overview)  
2. [Dataset](#dataset)  
3. [Model Architecture](#model-architecture)  
4. [Dependencies](#dependencies)  
5. [Code Description](#code-description)  
6. [Usage Instructions](#usage-instructions)  
7. [Results and Performance](#results-and-performance)  

---

## Overview

Brain tumor classification is a critical task in the medical field, where timely and accurate detection can save lives. This project utilizes deep learning techniques to automate the classification process. The CNN model processes MRI images, extracting spatial features and learning patterns to distinguish between the presence and absence of a tumor.

---

## Dataset

The dataset used for this project consists of MRI images organized into two folders:
- `yes`: Images of brain scans with tumors.
- `no`: Images of brain scans without tumors.

### Dataset Structure:
Brain MRI/
│
├── yes/
│   ├── Y1.jpg
│   ├── Y2.jpg
│   └── ...
│
└── no/
    ├── N1.jpg
    ├── N2.jpg
    └── ...

---

## Model Architecture

The CNN (`SimpleConvNet`) consists of:
- **Convolutional Layer**: Extracts spatial features from the image.
- **MaxPooling Layer**: Reduces spatial dimensions while preserving key information.
- **Fully Connected Layers**: Maps extracted features to class probabilities.

Input images are resized to \(224 \times 224\) pixels and converted to grayscale. The model is trained using `CrossEntropyLoss` and optimized with `Adam`.

---

## Dependencies

To run this project, the following Python libraries are required:
- `torch`
- `torchvision`
- `PIL`
- `pandas`
- `scikit-learn`
- `pickle`

Install them using:
```bash
pip install torch torchvision pillow pandas scikit-learn

---

## Code Description

### Key Features:
1. **Dataset Preparation**:
   - A custom PyTorch Dataset (`ImageDataset`) is implemented to load and preprocess the images. 
   - Image paths and labels are stored in a Pandas DataFrame, and labels are mapped to numeric indices using `label_mapping`.
   - The mapping is saved for later inference.

2. **Training**:
   - The CNN model (`SimpleConvNet`) processes grayscale images resized to \(224 \times 224\).
   - The training loop includes forward passes, loss calculation using `CrossEntropyLoss`, and backpropagation with the `Adam` optimizer.
   - Model weights are saved after training for future use.

3. **Inference**:
   - A separate function loads the trained model and predicts the class of a new image.
   - The predicted numeric class is mapped back to the corresponding label using the saved `label_mapping`.

---

## Usage Instructions

1. **Prepare the Dataset**:
   - Ensure the dataset is structured with `yes` and `no` folders containing corresponding MRI images.

2. **Install Dependencies**:
   - Install the required Python libraries:
     ```bash
     pip install torch torchvision pillow pandas scikit-learn
     ```

3. **Train the Model**:
   - Update the `dataset_path` variable in the code to point to your dataset location.
   - Run the script to train the model:
     ```bash
     python train_model.py
     ```

4. **Save Model and Labels**:
   - The trained model weights will be saved as `model_weights.pth`.
   - The label mapping is stored as `label_mapping.pkl`.

5. **Make Predictions**:
   - Use the `predict()` function to classify new MRI images:
     ```bash
     python predict_image.py --image_path <path_to_image>
     ```

---

## Results and Performance

### Training Loss:
The training process shows consistent learning with the following epoch-wise losses:

```plaintext
Epoch [1/10], Loss: 0.6843
Epoch [2/10], Loss: 0.6899
Epoch [3/10], Loss: 0.6837
...
Epoch [10/10], Loss: 0.6879
