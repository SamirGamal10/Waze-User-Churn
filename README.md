# Waze User Churn Prediction 🚗📊

## 📌 Project Overview
This project focuses on analyzing Waze user behavior and predicting user churn using machine learning techniques.  
The goal is to identify users who are likely to stop using the application and understand the key factors behind churn.

---

## 📂 Dataset
- Source: Waze user activity dataset
- Target Variable: `label`
  - `0` → Retained User
  - `1` → Churned User
- Features include:
  - User activity metrics
  - Device type
  - Usage patterns
  - Engagement statistics

---

## 🛠️ Technologies & Tools
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

---

## 🔍 Project Workflow
1. Data Loading & Exploration  
2. Data Cleaning & Missing Value Handling  
3. Outlier Detection & Treatment (IQR Method)  
4. Feature Encoding & Scaling  
5. Handling Class Imbalance using SMOTE  
6. Model Training & Evaluation  
7. Model Comparison  

---

## 🤖 Machine Learning Models Used
- Bagging Classifier (KNN-based)
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier

---

## 📈 Model Evaluation Metrics
- Accuracy
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## ✅ Results
Ensemble models showed strong performance, with **Random Forest** and **Gradient Boosting** achieving the highest accuracy and ROC-AUC scores.