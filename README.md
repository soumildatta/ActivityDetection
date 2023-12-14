# ActivityDetection
# Human Activity Recognition Using Smartphone Sensors

## Overview

This project focuses on human activity recognition using data collected from smartphone sensors. The goal is to classify activities into categories such as walking, standing, sitting, etc., by analyzing time and frequency domain features derived from accelerometer and gyroscope data.

## Dataset

This project utilizes the popular [HAR dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) from UC Irvine. The dataset comprises readings from smartphone sensors, consolidated into two main files: train.csv and test.csv. Each record includes sensor measurements across multiple dimensions, with a total of 563 columns.

## Features

The sensor data is categorized into time-domain (tBodyAcc, tBodyGyro, etc.) and frequency-domain (fBodyAcc, fBodyGyro, etc.) features. The dataset also includes labels for various activities like standing, sitting, walking, etc.

## Methodology

1. Data Exploration and Visualization: Initial analysis of the sensor data and activities.
2. Feature Engineering: Selection of key features using techniques like SelectKBest.
3. Dimensionality Reduction: Application of PCA to reduce feature dimensions while preserving variance.
4. Data Preprocessing: Label encoding of categorical variables and standardization of feature scales.
5. Model Development: Implementation of various classifiers such as Random Forest, SVM, XGBoost, and MLP Classifier.
6. Model Evaluation: Accuracy assessment, confusion matrix visualization, and classification reports.
7. Hyperparameter Tuning: Using GridSearchCV to optimize model parameters.

## Models Used

- Random Forest Classifier
- Support Vector Classifier (SVC)
- XGBoost Classifier
- Multilayer Perceptron Classifier (MLP)

## Results

The project demonstrates the effectiveness of machine learning algorithms in classifying human activities based on sensor data. Post hyperparameter tuning, significant improvements in model performance were observed.

## Requirements

- Python 3.x
- __Libraries:__ pandas, numpy, matplotlib, seaborn, sklearn, xgboost

## Usage

1. Clone the repository.
2. Install the required libraries: pip install -r requirements.txt
3. Run the Jupyter Notebook to train models and evaluate performance.

## Contribution

Contributions to improve the model performance or to propose new approaches are welcome. Please feel free to fork the repo and create pull requests.