# WIE2003 Introduction to Data Science Project - Data Driven Property Price Analysis


## 📍 Project Overview
Analyzed Kuala Lumpur residential transactions to build a classification model that predicts property price categories (Budget to Luxury) based on location and physical features.

## 🛠️ Data Pipeline
* **Cleaning:** Handled nulls (grounded properties = 0), unified area metrics into a single **Effective Area**, and converted currency strings to floats.
* **Standardization:** Converted all units to **sqft** ($1m^2 = 10.7639sqft$) and removed price/size outliers.
* **EDA:** Visualized trends in tenure (Freehold vs. Leasehold), popularity by district, and price-per-sqft distributions.

## 🤖 Machine Learning
* **Engineering:** One-Hot Encoding for property types; Target Encoding for high-cardinality districts (Mukim); Scaled numerical features with `StandardScaler`.
* **Models:** Compared **Logistic Regression**, **Random Forest**, and **XGBoost**.
* **Validation:** Used **Stratified 80/20 Split** and **K-Fold Cross-Validation** to ensure consistent accuracy across all price brackets.
* **Tuning:** Optimized hyperparameters using `RandomizedSearchCV`.

## 📂 Tech Stack
* **Language:** Python (Pandas, NumPy)
* **Visuals:** Matplotlib, Seaborn
* **ML:** Scikit-Learn, XGBoost
* **Environment:** Google Colab & Google Drive

---

## 🚀 Streamlit Setup Guide

## 📌 Requirements
Make sure you have:
- Python **3.9 or above** (recommended: 3.10 / 3.11)
- pip (latest version recommended)

Check versions:
```bash
python --version
pip --version
```
---

📦 Install Dependencies
```bash
pip install -r requirements.txt
```

📌 If Streamlit is not installed:
```bash
pip install streamlit
```

▶️ Run the Application
```bash
python3 -m streamlit run app.py
```

Then open in your browser:
http://localhost:8501
---
