# Credit Card Fraud Detection

This project focuses on building machine learning models to detect fraudulent credit card transactions. Given the imbalanced nature of the dataset, we evaluate multiple models and assess their performance in identifying fraud.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction
Credit card fraud is a growing concern in the financial industry, leading to billions in losses each year. Timely and accurate detection of fraudulent transactions is essential to prevent financial loss and ensure security. In this project, we aim to develop and evaluate machine learning models for fraud detection using various features from transactional data.

## Dataset
The dataset includes transaction records with the following key features:
- **cc_num**: Credit card number
- **amt**: Transaction amount
- **zip**: Zip code
- **lat**: Latitude
- **long**: Longitude
- **city_pop**: City population
- **unix_time**: Timestamp of the transaction
- **merch_lat**: Merchant latitude
- **merch_long**: Merchant longitude
- **is_fraud**: Target label (1 for fraudulent transactions, 0 for non-fraudulent)

### Dataset Split
The dataset is split into training and testing sets using the `train_test_split` method from `sklearn`. The training set is used to train the models, and the test set is used to evaluate their performance.

## Preprocessing
Before training the models, the following preprocessing steps are applied:
- **Label Encoding**: Categorical variables are encoded using label encoding to convert them into a format suitable for machine learning algorithms.
- **Data Splitting**: The dataset is split into training and testing sets.

## Models Used
We implemented the following machine learning models for fraud detection:

1. **Logistic Regression**: A linear model for binary classification that serves as our baseline.
   
   ```python
   log_model = LogisticRegression(max_iter=1000)
   log_model.fit(X_train, y_train)
   y_pred = log_model.predict(X_test)

   Decision Tree Classifier: A non-linear model that creates decision rules based on the features to classify data points.

   tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)

Random Forest Classifier: An ensemble model of decision trees, which reduces overfitting and improves prediction accuracy.

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


Evaluation Metrics
The following evaluation metrics are used to assess the models:

Accuracy Score: The percentage of correctly classified transactions.
Confusion Matrix: A matrix showing the number of true positives, false positives, true negatives, and false negatives.
Classification Report: A detailed report showing precision, recall, and F1-score for each class.

## Results

### Random Forest Classifier
Accuracy: To be completed after running the Random Forest model


Confusion Matrix: To be completed after running the Random Forest model


Precision (Fraud Class): To be completed after running the Random Forest model


Recall (Fraud Class): To be completed after running the Random Forest model


F1-Score (Fraud Class): To be completed after running the Random Forest model


The Random Forest Classifier is expected to perform even better, reducing overfitting and improving fraud detection by averaging multiple decision trees.

## Conclusion
Logistic Regression, while providing high overall accuracy, fails to detect fraudulent transactions effectively due to class imbalance.


Decision Tree Classifier provides a balanced performance with a significant improvement in fraud detection.


The Random Forest Classifier is expected to further improve performance by averaging across multiple decision trees.


## Future Work
Handling Class Imbalance: Techniques like SMOTE (Synthetic Minority Over-sampling Technique) or undersampling the majority class can be used to address the class imbalance and improve fraud detection for Logistic Regression.


Feature Engineering: Additional features like transaction patterns or behavioral analysis could improve model performance.


Hyperparameter Tuning: Optimizing the parameters of the models (e.g., max depth of the decision tree, number of estimators in Random Forest) can further enhance accuracy and fraud detection rates.
vbnet

## Dataset 

One can view the dataset through my Gmail 

https://drive.google.com/file/d/1oyo7mui9w2NaiJ6ld4qLpLiObFhFehKU/view?usp=sharing

https://drive.google.com/file/d/1CYebfl1JW6Zstuo_WUKjalbRLF1eT06k/view?usp=sharing
