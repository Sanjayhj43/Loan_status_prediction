# Loan_status_prediction
Predicting Loan Status
# Loan Approval Prediction Project

# Overview
Welcome to the Loan Approval Prediction Project! This project is designed to predict whether a loan application will be approved or not based on a set of applicant features. We achieve this using a machine learning model, specifically a Support Vector Machine (SVM) classifier. Through data preprocessing, label encoding, and careful model training, we aim to provide accurate loan approval predictions.

# Table of Contents
Dataset - https://www.kaggle.com/datasets/ninzaami/loan-predication
Installation of required Libraries - Pandas, Numpy, Seaborn, Sklearn and suitable models
Usage - Train test split methods to train the data
Model - Support vector machine model
Results - Predicts Loan status

# Dataset
Our dataset contains essential information about loan applicants, including their income, credit history, marital status, gender, and more. To ensure accurate predictions, we have carefully handled missing values and used label encoding to convert categorical variables into numerical form.

# Installation
Clone this repository to your local machine.

Install the required Python libraries listed in the requirements.txt file:


pip install -r requirements.txt


# Usage
After installing the required libraries, you can run the loan_approval_prediction.py script.
The script performs essential tasks, such as preprocessing the data, splitting it into training and testing sets, and training the SVM classifier.
After training, the script evaluates the model's accuracy, precision, recall, and F1-score on the testing data.
Finally, the script provides loan approval predictions for new data.

# Model
In this project, we employ a Support Vector Machine (SVM) classifier to make accurate loan approval predictions. SVM is a powerful algorithm that effectively separates data into distinct classes based on the provided features. Through careful hyperparameter tuning and model training, we ensure the SVM's optimal performance.

# Results
Our trained SVM classifier achieved an accuracy of approximately 80% on the training data and an impressive 84.5% on the testing data. These results highlight the model's effectiveness in predicting loan approval outcomes.
