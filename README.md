# 📉 Customer Churn Prediction — IBM Telco Dataset

## Overview
Built a machine learning pipeline to predict customer churn for a telecom 
company using the IBM Telco dataset (7,043 customers). Compared XGBoost 
against a Logistic Regression baseline with SMOTE oversampling to handle 
class imbalance.

## Key Results
| Model | F1 Score |
|---|---|
| Logistic Regression (baseline) | 0.61 |
| XGBoost (tuned + SMOTE) | **0.63+** |

- Applied SMOTE to balance the training set (26% churn rate)
- Threshold tuning to maximise F1 score
- SHAP values used to identify top churn drivers
- Top churn driver: **Contract type (Two Year)**

## Visualisations
![EDA](churn_eda.png)
![Model Comparison](model_comparison.png)
![SHAP](shap_plot.png)

## Dataset
- **Source:** IBM Telco Customer Churn (public dataset)
- **Size:** 7,043 customers, 21 features → 30 after encoding
- **Target:** Churn (Yes/No)

## Techniques Used
- Exploratory Data Analysis
- Feature engineering & one-hot encoding
- SMOTE oversampling for class imbalance
- XGBoost with hyperparameter tuning
- Threshold optimisation
- SHAP explainability

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

## How to Run
1. Open `customer_churn.ipynb` in Google Colab or Jupyter
2. Run all cells top to bottom
3. Dataset loads automatically from URL — no download needed

## Author
Jayapradha Perinparasa — MSc Data Science, University of Hertfordshire  
[LinkedIn](https://www.linkedin.com/in/jayapradhap) | 
[Research Paper](https://doi.org/10.31705/ADScAI.2025.08)
