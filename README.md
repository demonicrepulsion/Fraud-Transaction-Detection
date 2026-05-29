# Fraud Transaction Detection System

## Overview

A Machine Learning project that detects fraudulent transactions using classification techniques. The project includes data generation, exploratory data analysis, model training, evaluation, visualization, model persistence, and real-time prediction.

## Features

- Synthetic Fraud Dataset Generation
- Exploratory Data Analysis (EDA)
- Logistic Regression Model
- Random Forest Model
- Confusion Matrix Visualization
- Feature Importance Analysis
- Model Saving using Joblib
- Real-Time Fraud Prediction

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Jupyter Notebook

## Project Structure

```text
Fraud-Transaction-Detection
│
├── data
├── notebooks
├── src
├── models
├── images
├── README.md
├── requirements.txt
└── .gitignore
```

## Models Implemented

### Logistic Regression
- Accuracy: 96.05%

### Random Forest
- Accuracy: 100%

## Visualizations

### Confusion Matrix
See:
images/confusion_matrix.png

### Feature Importance
See:
images/feature_importance.png

## Real-Time Prediction

The system loads a trained model and predicts whether a transaction is fraudulent or legitimate.

## Future Improvements

- XGBoost Implementation
- SHAP Explainability
- Streamlit Web App
- Real Credit Card Fraud Dataset Integration