# **Breast Cancer Classification Project**

This project implements a **Breast Cancer Classification** system using machine learning techniques. It uses the **Breast Cancer Wisconsin (Diagnostic) dataset** to classify tumors as either malignant or benign. The current implementation uses the **AdaBoost Classifier** for this purpose, but other algorithms can also be tested and integrated.

## **Overview**

Breast cancer is one of the most common types of cancer affecting women worldwide. Early detection and accurate classification of breast tumors as benign or malignant are crucial for effective treatment. This project demonstrates how machine learning models can assist in such classification tasks.

The project loads the Breast Cancer dataset from the popular `sklearn` library, preprocesses the data (standardization), splits it into training and testing sets, and trains a classifier to make predictions. Model performance is evaluated using accuracy, confusion matrix, and classification reports.

In future iterations, this project will be integrated into a **Streamlit** web application for easy accessibility and interactive usage.

## **Features**

- Uses **AdaBoostClassifier** to classify breast cancer as malignant or benign
- Displays the confusion matrix and classification report for performance evaluation
- Predicts tumor type for new input data samples
- Preprocesses the dataset with **standardization** (mean = 0, variance = 1)
- Future plans to include a web interface using **Streamlit**

## **Dependencies**

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `scikit-learn`

You can install them using `pip`:

```bash
pip install pandas numpy scikit-learn
