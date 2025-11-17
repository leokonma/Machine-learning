

# 🧠 Machine Learning — Player Performance Analytics

## 📋 Overview

This project applies **Machine Learning**, **feature engineering**, **unsupervised analysis**, and **supervised prediction** to model football (soccer) player performance.

It includes:

* 🧼 Data cleaning & structuring
* ⚙️ Advanced feature enrichment (per90, lags, deltas, z-scores, weighted metrics)
* 📉 Dimensionality reduction (PCA, LASSO, RF importance)
* 🎯 Clustering for player segmentation
* 🔮 Supervised learning for Ballon d’Or–style predictions

---

## ⚙️ Project Workflow

### 1️⃣ Data Acquisition

`src/a_data_download.py`

### 2️⃣ Data Cleaning

`src/data_cleaning.py`

### 3️⃣ Feature Enrichment

`src/data_enrichment.py`

### 4️⃣ Unsupervised Analysis

Located in `/unsupervised methods/`.

Includes PCA, LASSO, RF importance, and clustering (K-means, hierarchical, DBSCAN).

### 5️⃣ Supervised Analysis (NEW)

Located in `/supervised_methods/`.

Full prediction pipeline:

| Notebook                              | Purpose                                                                                    |
| ------------------------------------- | ------------------------------------------------------------------------------------------ |
| **01_build_supervised_dataset.ipynb** | Build the training dataset (labels, target engineering, lag handling, leakage prevention). |
| **02_benchmark_models.ipynb**         | Compare Logistic, Random Forest, KNN, SVM, XGBoost (baseline metrics).                     |
| **03_model_trainin.ipynb**            | Train the best-performing model (currently Random Forest).                                 |
| **04_hyperparameter_tuning_rf.ipynb** | Grid-search, randomized search, Bayesian optimization (depending on config).               |
| **05_test_prediction.ipynb**          | Predict on unseen seasons (e.g., 2025), evaluate model generalization.                     |
| **prediction.ipynb**                  | Final standalone predictor for deployment-style usage.                                     |

And stored models:

* `rf_baseline.pkl`
* `rf_best_tuned.pkl`
* `rf_final_2008_2022.pkl`

---

## 📂 Repository Structure (Updated)

```
Machine-learning/
│
├── data/
│   ├── raw/
│   ├── clean/
│   └── enriched/
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
│   ├── 01_build_supervised_dataset.ipynb
│   ├── 02_benchmark_models.ipynb
│   ├── 03_model_trainin.ipynb
│   ├── 04_hyperparameter_tuning_rf.ipynb
│   ├── 05_test_prediction.ipynb
│   ├── prediction.ipynb
│   ├── rf_baseline.pkl
│   ├── rf_best_tuned.pkl
│   └── rf_final_2008_2022.pkl
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

```powershell
pip install -r requirements.txt
```

## 4️⃣ Run data pipelines

```powershell
python -m src.data_cleaning
python -m src.data_enrichment
```

## 5️⃣ Run notebooks

In VS Code or Jupyter.

---

## 🔮 Supervised Learning Info (NEW)

The supervised pipeline generates a classification target based on seasonal performance and predicts the probability of a high-impact award-level season.

Key ML components:

* Model evaluation: ROC-AUC, recall@k, precision, confusion matrix
* Hyperparameter tuning for Random Forest
* Full-year prediction for **season 2025**
* Stored model artifacts (.pkl) for reproducibility

---

## 🧩 Tech Stack

| Category        | Tools                                         |
| --------------- | --------------------------------------------- |
| Data            | pandas, numpy                                 |
| ML              | scikit-learn, xgboost (optional), statsmodels |
| Visualization   | seaborn, matplotlib                           |
| Dev             | VS Code, Jupyter                              |
| Version Control | Git + GitHub                                  |


## 👤 Author

**Leonardo Sánchez Castillo**
Data Analyst & Machine Learning Student

---
