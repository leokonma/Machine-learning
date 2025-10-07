# ⚽ Machine Learning Project: Football Player & Team Data

## 📊 Dataset Overview
This project uses multiple datasets covering **players**, **teams**, **performances**, **market values**, and **transfers**.  
Together, they allow an in-depth analysis of player careers and team dynamics.

The available dataframes are:

- **`player_injured_df`** → Records of player injuries, including dates and types.  
- **`player_latest_market_value_df`** → Latest known market value of each player.  
- **`player_performances_df`** → Club-level performances such as matches, goals, and assists.  
- **`team_competitions_seasons_df`** → Teams’ participation in competitions by season.  
- **`team_details_df`** → General metadata about teams (name, country, founding year, etc.).  

---

## 📂 Data Storage
The raw data is **not stored in this repository** to keep it lightweight.  
Instead, all files are stored in **Google Drive**.  

To ensure reproducibility, we provide a script called **`download_data.py`** that automatically downloads all necessary folders.

---

## 📥 How to Access the Data

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
2. run the (`download_data.py`) file 
3. run the (`Data_module.py`) file 
4. you are able now to run Model_Module.ipynb choosing the kernel (.venv)

## 📂 Data Storage
The raw data is **not stored in this repository** (to keep it lightweight).  
Instead, all files live in **Google Drive**.

To ensure reproducibility, we provide a script (`download_data.py`) that downloads all necessary folders automatically.
## 🧠 Project Structure

- **📊 _reports/** → Contains detailed statistical reports before and after data cleaning (mean, median, missing values, etc.), as well as the optimized code version from the first project submission.  
- **📈 plots/** → Stores all visualizations (histograms, bar charts, distributions, etc.) used to illustrate trends and insights.  
- **🧹 Data_Module.py** → Core script responsible for cleaning, transforming, and enriching the raw data, preparing the final datasets for modeling.  
- **🤖 Model_Module.ipynb** → Jupyter Notebook for building, training, and evaluating machine learning models, including feature engineering and model performance analysis.  
- **📦 requirements.txt** → Lists all Python dependencies required to run the project. Install them using `pip install -r requirements.txt`.  
- **🚫 .gitignore** → Specifies files and folders excluded from version control (temporary files, virtual environments, datasets, etc.).  

## Notes:

The project is compatible with Python 3.9+.
Make sure to activate your virtual environment before running the scripts:

source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows

*in case of being interest in the 1st deliver a template of the old code, revised by chat gpt is included in the report files, that code is not longer in the repo due to changes in the structure including modules*