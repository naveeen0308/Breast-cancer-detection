**Breast Cancer Classification Project**

This project implements a Breast Cancer Classification system using machine learning techniques. It uses the Breast Cancer Wisconsin (Diagnostic) dataset to classify tumors as either malignant or benign. The current implementation uses the AdaBoost Classifier for this purpose, but other algorithms can also be tested and integrated.

**Overview**
Breast cancer is one of the most common types of cancer affecting women worldwide. Early detection and accurate classification of breast tumors as benign or malignant are crucial for effective treatment. This project demonstrates how machine learning models can assist in such classification tasks.

The project loads the Breast Cancer dataset from the popular sklearn library, preprocesses the data (standardization), splits it into training and testing sets, and trains a classifier to make predictions. Model performance is evaluated using accuracy, confusion matrix, and classification reports.

In future iterations, this project will be integrated into a Streamlit web application for easy accessibility and interactive usage.

**Features**
Uses AdaBoostClassifier to classify breast cancer as malignant or benign
Displays the confusion matrix and classification report for performance evaluation
Predicts tumor type for new input data samples
Preprocesses the dataset with standardization (mean = 0, variance = 1)
Future plans to include a web interface using Streamlit

**Dependencies**
The following Python libraries are required to run the project:
pandas
numpy
scikit-learn

**Dataset**
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Dataset, available through the sklearn.datasets.load_breast_cancer() function. The dataset contains 30 features extracted from breast tumor cell nuclei measurements.

Features: 30 real-valued features (e.g., radius, texture, perimeter, area, etc.)
Labels: Binary labels representing tumor classification:
0: Malignant
1: Benign

**How to Run**
Clone this repository or download the script to your local machine.
Install the required dependencies as mentioned above.
Run the Python script: python breast_cancer_classification.py
