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
  
   - ## 📊 Results and Visualizations

###  Model Performance Comparison

#### 🔹 Accuracy vs Loss
![Accuracy vs Loss](https://github.com/rayen-feb/machine_learning_breast_cancer_detection/raw/main/results/plots/accuracy_vs_loss.png)

#### 🔹 Confusion Matrix
![Confusion Matrix](https://github.com/rayen-feb/machine_learning_breast_cancer_detection/raw/main/results/plots/confusion_matrix.png)

#### 🔹 ROC Curves
![ROC Curve](https://github.com/rayen-feb/machine_learning_breast_cancer_detection/raw/main/results/plots/roc_curve.png)

These visualizations provide insight into model effectiveness and help compare performance across methods like Logistic Regression, SVM, and Neural Networks.

---

## Getting Started

### 1. Clone the repository
 
git clone https://github.com/rayen-feb/machine_learning_breast_cancer_detection-.git
cd machine_learning_breast_cancer_detection

### 2. Install dependencies
pip install -r requirements.txt

### 3.prepare data
Place raw dataset files in data/raw/
Preprocess data using notebooks or scripts/data_preprocessing.py
Save processed data in data/processed/

### 4. Train Models
Use the notebooks to train and evaluate each model
Metrics computed: Accuracy, Precision, Recall, F1-score

### 5. Deploy (Optional)
Run app.py to predict new patient data
Can be extended to a web interface or API


