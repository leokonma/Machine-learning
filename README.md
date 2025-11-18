
# 🧠 Machine Learning — Player Performance Analytics

### *Feature Engineering · Dimensionality Reduction · Clustering · Award Prediction*

---

## 📋 Overview

This project applies **Machine Learning**, **advanced feature engineering**, **unsupervised analysis**, and **supervised prediction** to analyze football (soccer) player performance across multiple seasons.

It includes:

* 🧼 **Data cleaning** & structuring
* ⚙️ **Advanced feature enrichment** (per90, lags, deltas, z-scores, weighted metrics, UCL strength)
* 📉 **Dimensionality reduction** (Correlation pruning, PCA, LASSO, Random Forest importance)
* 🎯 **Clustering** for player segmentation
* 🔮 **Supervised learning** for Balon d'Or–style rare-event prediction
* 🏆 **Prediction for the 2025 season**

---

## ⚙️ Project Workflow

### 1️⃣ Data Acquisition

`src/a_data_download.py`
Downloads and stores raw datasets (2008–2025).

### 2️⃣ Data Cleaning

`src/data_cleaning.py`
Cleans, normalizes, merges, and validates raw data.

### 3️⃣ Feature Enrichment

`src/data_enrichment.py`
Generates enriched features:
per90 metrics · lags · deltas · z-scores · weighted metrics · trophies · UCL progression · team strength.

---

## 🔍 Unsupervised Analysis

📁 Located in `/unsupervised methods/`

Includes:

* Correlation Analysis
* PCA Dimensionality Reduction
* LASSO Feature Selection
* Random Forest Feature Importance
* K-means Clustering
* Hierarchical Clustering
* Density-based Clustering (DBSCAN)

These notebooks support **feature selection** and **player archetype discovery**.

---

# 🤖 Supervised Learning Pipelines (UPDATED)

The project contains **two fully separated supervised pipelines**:

---

# ⭐ 1. **FINAL PIPELINE — Logistic Regression (Feature-Selected)**

📁 Located in `/supervised_methods/`

This is the **official and recommended** prediction pipeline.

| Notebook                                 | Purpose                                                        |
| ---------------------------------------- | -------------------------------------------------------------- |
| **01_lr_build_supervised_dataset.ipynb** | Build ML dataset (labels, target engineering, leak-safe lags). |
| **02_lr_benchmark_models.ipynb**         | Compare LR, RF, XGB, SVM, KNN on selected features.            |
| **03_lr_model_trainin.ipynb**            | Train Logistic Regression baseline (ElasticNet).               |
| **04_lr_hyperparameter_tuning_lr.ipynb** | RandomizedSearchCV tuning for LR (C, l1_ratio).                |
| **05_lr_test_prediction.ipynb**          | Evaluate on seasons 2023–2024 (out-of-sample test).            |
| **prediction.ipynb**                     | **Final prediction for season 2025**.                          |

### 🔎 Key characteristics:

* Uses **only final_features** (30 best features selected by Corr + RF importance).
* **Pipeline architecture:**
  `ColumnTransformer` → `StandardScaler` + `OneHotEncoder` → `SMOTE` → `Logistic Regression`.
* **Zero data leakage** across all stages.
* Winner detection for Ballon d’Or-style rare events.
* Stored model: `lr_tuned_pipeline.pkl`
* **Best model by AUC (≈ 0.99978)** and best calibration for probability ranking.

👉 **This is the main model used for predicting the top candidates in 2025.**

---

# 🌲 2. **LEGACY PIPELINE — Random Forest (All Features)**

📁 Located in `/supervised_methods/`

These notebooks represent your **original approach**, preserved for transparency:

| Notebook                              | Purpose                                                  |
| ------------------------------------- | -------------------------------------------------------- |
| **06_build_supervised_dataset.ipynb** | Early version of dataset builder (no feature selection). |
| **07_benchmark_models.ipynb**         | Benchmark using all features.                            |
| **08_model_trainin.ipynb**            | Train Random Forest baseline.                            |
| **09_hyperparameter_tuning_rf.ipynb** | RandomizedSearchCV tuning for RF.                        |
| **010_test_prediction.ipynb**         | RF final evaluation / alternative prediction.            |

### Characteristics:

* Uses **all raw enriched features** (no dimensionality reduction).
* Includes **RF hyperparameter tuning** (n_estimators, depth, etc.).
* Serves as **comparison baseline** to validate improvements.
* Stored RF artifacts:
* `rf_best_tuned.pkl`
  

👉 *Useful for understanding model evolution, but **not recommended** for final deployment.*

---

# 📂 Repository Structure (Updated)

```
Machine-learning/
│
├── data/
│   ├── raw/            ← original downloaded datasets
│   ├── clean/          ← cleaned intermediate datasets
│   └── enriched/       ← enriched with lags/deltas/UCL strength
│
├── src/
│   ├── __init__.py
│   ├── a_data_download.py
│   ├── data_cleaning.py
│   ├── data_enrichment.py
│   └── utils.py
│
├── unsupervised methods/
│   ├── Correlation_Analysis.ipynb
│   ├── PCA_Dim_Reduc.ipynb
│   ├── Lasso_Dim_Reduc.ipynb
│   ├── Random_Forest_Dim_Reduc.ipynb
│   ├── k-means_Clustering.ipynb
│   ├── Hierarchical_Clustering.ipynb
│   └── Density_Clustering.ipynb
│
├── supervised_methods/
│   ├── 01_lr_build_supervised_dataset.ipynb
│   ├── 02_lr_benchmark_models.ipynb
│   ├── 03_lr_model_trainin.ipynb
│   ├── 04_lr_hyperparameter_tuning_lr.ipynb
│   ├── 05_lr_test_prediction.ipynb
│   ├── lr_tuned_pipeline.pkl     ← final model
│   │
│   ├── 06_build_supervised_dataset.ipynb
│   ├── 07_benchmark_models.ipynb
│   ├── 08_model_trainin.ipynb
│   ├── 09_hyperparameter_tuning_rf.ipynb
│   ├── 010_test_prediction.ipynb
│   └── prediction.ipynb <- legacy prediction           
│
├── requirements.txt
└── README.md
```

---

# 🚀 How to Run the Project

## 1️⃣ Clone the repository

```bash
git clone https://github.com/leokonma/Machine-learning.git
cd Machine-learning
```

## 2️⃣ (Windows PowerShell) Enable venv

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run data pipelines

```bash
python -m src.data_cleaning
python -m src.data_enrichment
```

## 5️⃣ Run notebooks (VS Code or Jupyter)

---

# 🔮 Supervised Learning Summary

### Logistic Regression Pipeline (FINAL):

* Best model for rare-event prediction
* Best AUC and probability calibration
* Feature-selected
* Predicts 2025 final candidates

### Random Forest Pipeline (LEGACY):

* First-generation pipeline
* Uses all features
* Serves as baseline comparison

---

# 🧩 Tech Stack

| Category        | Tools                                   |
| --------------- | --------------------------------------- |
| Data            | pandas, numpy                           |
| ML              | scikit-learn, imbalanced-learn, xgboost |
| Visualization   | seaborn, matplotlib                     |
| Dev             | VS Code, Jupyter                        |
| Version Control | Git + GitHub                            |

---

## 👤 Author

**Leonardo Sánchez Castillo**
Data Analyst & Machine Learning Student

---
