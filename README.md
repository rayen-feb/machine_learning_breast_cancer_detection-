# Breast Cancer Detection using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-green)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Project Overview
This project implements multiple machine learning models to predict breast cancer (benign vs malignant) using structured clinical and physiological data. The main goal is to compare different modeling approaches, maximize predictive accuracy, and maintain interpretability for medical analysis.

Key features of this project:
- Data preprocessing and feature exploration
- Multiple ML models for classification
- Model evaluation and comparison
- Optional deployment via `app.py`

---

## Dataset
The dataset includes clinical measurements from adult patients. Key features:

- Age  
- Body Mass Index (BMI)  
- Percent Body Fat  
- Hand Grip Strength (HGS)  
- Flexibility (Sit-and-reach)  
- Muscular Endurance (Sit-ups)  
- Additional clinical measurements  

**Target variable:** `Diagnosis` (`Benign` or `Malignant`)

> ⚠️ The dataset itself is **not included** in this repository due to size and privacy.  
> Place raw data in `data/raw/` and processed data in `data/processed/`.

---

## Project Structure
machine_learning_breast_cancer_detection/
│
├── data/ # Raw and processed datasets
│ ├── raw/
│ └── processed/
│
├── notebooks/ # Jupyter notebooks
│ ├── 1_data_preprocessing.ipynb
│ ├── 2_model_GRU_SVM.ipynb
│ ├── 3_model_MLP.ipynb
│ ├── 4_model_LinearRegression.ipynb
│ ├── 5_model_SVM.ipynb
│ ├── 6_model_SoftmaxRegression.ipynb
│ └── 7_model_L1_L2_NN.ipynb
│
├── scripts/ # Python scripts for reproducibility
│ ├── data_preprocessing.py
│ ├── train_models.py
│ └── utils.py
│
├── app.py # Optional deployment interface
├── requirements.txt # Python packages
├── README.md # This file
└── .gitignore # Ignore temporary/cache files

---

## Models Implemented

1. **Softmax (Multinomial) Logistic Regression**
   - Predicts class probabilities for binary classification.
   - Interpretable and suitable for small datasets.

2. **Support Vector Machines (SVM)**
   - Linear SVM: assumes linearly separable features.
   - Non-linear SVM: uses kernels (RBF/Polynomial) for complex patterns.

3. **Multi-Layer Perceptron (MLP)**
   - Feedforward neural network with multiple layers, ReLU activations, and dropout to prevent overfitting.

4. **GRU + SVM Hybrid**
   - GRU captures sequential embeddings from tabular features.
   - SVM uses these embeddings for classification.

5. **Neural Networks with L1/L2 Regularization**
   - Reduces overfitting using weight penalties.
   - Tested multiple architectures for optimal hyperparameters.

6. **Nearest Neighbors (NN1 & NN2)**
   - Euclidean and Manhattan distance metrics.
   - Suitable for tabular datasets with structured features.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/rayen-feb/machine_learning_breast_cancer_detection-.git
cd machine_learning_breast_cancer_detection
### 2. install dependencies

---

## Models Implemented

1. **Softmax (Multinomial) Logistic Regression**
   - Predicts class probabilities for binary classification.
   - Interpretable and suitable for small datasets.

2. **Support Vector Machines (SVM)**
   - Linear SVM: assumes linearly separable features.
   - Non-linear SVM: uses kernels (RBF/Polynomial) for complex patterns.

3. **Multi-Layer Perceptron (MLP)**
   - Feedforward neural network with multiple layers, ReLU activations, and dropout to prevent overfitting.

4. **GRU + SVM Hybrid**
   - GRU captures sequential embeddings from tabular features.
   - SVM uses these embeddings for classification.

5. **Neural Networks with L1/L2 Regularization**
   - Reduces overfitting using weight penalties.
   - Tested multiple architectures for optimal hyperparameters.

6. **Nearest Neighbors (NN1 & NN2)**
   - Euclidean and Manhattan distance metrics.
   - Suitable for tabular datasets with structured features.

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/rayen-feb/machine_learning_breast_cancer_detection-.git
cd machine_learning_breast_cancer_detection
### 2.install dependencies 
pip install -r requirements.txt
### 3.prepare data 
place raw dataset files in data/raw/
Preprocess data using notebooks or scripts/data_preprocessing.py
Save processed data in data/processed/

### 4.train models
Use the notebooks to train and evaluate each model.
Metrics computed: Accuracy, Precision, Recall, F1-score

### 5.deploy(optionally)
Run app.py to predict new patient data.
Can be extended to a web interface or API.

### results 
Each notebook includes performance metrics for each model.

Comparative evaluation highlights the best-performing model.

Focus on accuracy and interpretability, important for clinical applications.


