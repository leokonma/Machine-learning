# 🧠 Machine Learning — Player Performance Analytics

## 📋 Overview

This project applies **Machine Learning**, **feature engineering**, and **unsupervised methods** to analyze football (soccer) player performance.
It focuses on:

* 🧼 Data cleaning & structuring
* ⚙️ Automated feature enrichment (per90 stats, lag features, weighted metrics)
* 📉 Dimensionality reduction (PCA, LASSO, RF importance)
* 🎯 Clustering & player segmentation

---

## ⚙️ Project Workflow

### 1️⃣ Data Acquisition

`A_Data_Download.py` → Downloads datasets to `data/raw/`.

### 2️⃣ Data Cleaning

`src/data_cleaning.py`

### 3️⃣ Feature Enrichment

`src/data_enrichment.py`

### 4️⃣ Exploratory & Unsupervised Analysis (notebooks)

Located in `/unsupervised methods/`.

### 5️⃣ Dimensionality Reduction

PCA, LASSO, Random Forest notebooks.

### 6️⃣ Clustering

K-means, Hierarchical, DBSCAN.

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

---

## 2️⃣ Create & activate virtual environment (**Windows PowerShell**)

PowerShell blocks script execution by default, so first:

### 🟦 **Bypass PowerShell policy (safe, temporary)**

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This only affects the current PowerShell window — safe and recommended.

### 🟩 Create venv

```powershell
python -m venv .venv
```

### 🟩 Activate venv (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

You will see:

```
(.venv) PS C:\Users\...>
```

---

## 3️⃣ Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 4️⃣ Run data pipeline scripts

Use the `-m` flag so Python treats `src/` as a package:

```powershell
python -m src.data_cleaning
python -m src.data_enrichment
```

---

## 5️⃣ Run notebooks

Launch Jupyter:

```powershell
jupyter notebook
```

Or open the folder in VS Code and run the notebooks directly.

Imports work thanks to the automatic project-root resolver.

---

## 🧩 Tech Stack

| Category        | Tools                     |
| --------------- | ------------------------- |
| Languages       | Python 3.10+              |
| Data            | pandas, numpy             |
| ML              | scikit-learn, statsmodels |
| Visualization   | seaborn, matplotlib       |
| Environment     | VS Code, Jupyter          |
| Version Control | Git + GitHub              |

---

## 📈 Future Improvements

* Add supervised prediction models (Random Forest, XGBoost)
* Deploy dashboard with Streamlit or Dash
* Add SHAP or LIME interpretability
* MLOps CI/CD pipeline (GitHub Actions)

---

## 🧑‍💻 Author

**Leonardo Sánchez Castillo**
Data Analyst & Machine Learning Student

---

Si quieres, puedo añadir **badges**, **un logo**, o una **sección de troubleshooting** para errores comunes (PowerShell, imports, venv, etc.).
