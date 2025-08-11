

# PROJECT ARRHYTHMIA

## Disclaimer
This repository does not have any official medical affiliation.

## Introduction
This project focuses on predicting and classifying arrhythmia using various machine learning algorithms.
The dataset used for this project is from the:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
This data consists of 452 examples across 16 different classes. Among these, 245 examples are labeled as "Normal," while the remaining represent 12 different types of arrhythmias, including "Coronary artery disease" and "Right bundle branch block"

**Objectives**
The goal of this project is to predict whether a person suffers from arrhythmia, and if they do suffer from arrhythmia, classify the type of arrhythmia into one of the 12 available groups.

#### 🫡 You can help me by Donating
[![BuyMeACoffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/heytanix)

## Algorithms used in the project
 1. **K-Nearest Neighbors(KNN) Classifier**
 2. **Logistic Regression**
 3. **Decision Tree Classifier**
 4. **Linear Support Vector Classifier (SVC)**
 5. **Kernelized Support Vector Classifier (SVC)**
 6. **Random Forest Classifier**
 7. **Principal Component Analysis (PCA)** (For dimensionality reduction)

## Project workflow

### Step 1 : Data exploration
- Analyzed the 279 features to identify patterns and correlations that could help with the prediction.
- Addressed the challenge of the high number of features compared to the limited number of examples by employing PCA(Principal component analysis).

### Step 2 : Data pre-processing:
- Handled missing values, standardized data, and prepared it for machine learning models.
- Applied **Principal component analysis** to reduce the feature space and eliminate collinearity, improving both execution time and model performance.

### Step 3 : Model Training and evaluation
- Trained various machine learning algorithms on the dataset.
- Evaluated model performance using accuracy, recall and other relevant metrics.

### Step 4 : Model Tuning with PCA
- PCA helped reduce the complexity of the dataset, leading to improved model accuracy and reduced overfitting.
-  After applying PCA, models were retrained, and significant improvements were observed.

### Conclusion
Applying **Principal Component Analysis(PCA)** to the resampled data significantly improved the performance of the models. PCA works by creating non-collinear components that prioritize variables with high variance, thus reducing dimensionality and collinearity, which are key issues in large datasets. PCA not only enhanced the overall execution time but also improved the quality of predictions.

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [PCA Concepts](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)

---

This `README.md` offers clear documentation of the objectives, algorithms used, results, and the significance of PCA in your project. It also provides essential information on how to run the project and the prerequisites.