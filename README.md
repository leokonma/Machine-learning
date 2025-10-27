# 🧠 Machine Learning — Player Performance Analytics

## 📋 Overview
This project applies **machine learning and data analytics** to evaluate player performance in football (soccer).
It explores how different **dimensionality reduction** and **clustering techniques** can help identify player profiles and potential predictors of elite performance (e.g., *Ballon d’Or–level indicators*).

The workflow covers every stage of the data pipeline — from **data acquisition and cleaning**, to **feature enrichment, analysis, and visualization**.

---

## ⚙️ Project Workflow

1. **Data Acquisition**
   - `A_Data_Download.py` — Retrieves raw datasets and stores them locally.

2. **Data Cleaning**
   - `Data_Cleaning.py` — Cleans and structures the raw data.

3. **Feature Enrichment**
   - `Data_Enrichment.py` — Builds additional variables (per 90 metrics, ratios, lag features).

4. **Exploratory Analysis**
   - `Correlation_Analysis.ipynb` — Examines variable relationships and correlations.

5. **Dimensionality Reduction**
   - `PCA_Dim_Reduc.ipynb` — Principal Component Analysis.
   - `Lasso_Dim_Reduc.ipynb` — Feature selection using LASSO.
   - `Random_Forest_Dim_Reduc.ipynb` — Variable importance based on Random Forests.

6. **Clustering and Player Segmentation**
   - `k-means_Clustering.ipynb` — K-Means clustering.
   - `Hierarchical_ClusteringL.ipynb` — Hierarchical clustering.
   - `Density_Clustering.ipynb` — Density-based clustering.

---

## 📂 Repository Structure

```
Machine-learning-main/
├── .gitignore
├── A_Data_Download.py
├── Correlation_Analysis.ipynb
├── Data_Cleaning.py
├── Data_Enrichment.py
├── Density_Clustering.ipynb
├── Hierarchical_ClusteringL.ipynb
├── Lasso_Dim_Reduc.ipynb
├── PCA_Dim_Reduc.ipynb
├── Random_Forest_Dim_Reduc.ipynb
├── k-means_Clustering.ipynb
├── requirements.txt
```

---

## 🚀 How to Run

### 1️⃣ Clone the repository
```bash
git clone https://github.com/leokonma/Machine-learning.git
cd Machine-learning
```

### 2️⃣ Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Execute the pipeline
```bash
python A_Data_Download.py
python Data_Cleaning.py
python Data_Enrichment.py
```

Then open the analysis notebooks such as:
```
Random_Forest_Dim_Reduc.ipynb
k-means_Clustering.ipynb
```

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|-------|
| Languages | Python 3.10+ |
| Libraries | pandas, numpy, scikit-learn, matplotlib, seaborn |
| Environment | Jupyter Notebooks, Visual Studio Code |
| Version Control | Git / GitHub |

---

## 📈 Future Improvements
- Add predictive modeling for player performance forecasting.  
- Implement interactive dashboards with Plotly or Streamlit.  
- Automate reporting from pipeline outputs.

---

## 🧑‍💻 Author
**Leonardo Sánchez Castillo**  
Data Analyst & Machine Learning student  

