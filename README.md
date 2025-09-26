# ⚽ Machine Learning Project: Football Player & Team Data

## 📊 Dataset Overview
The dataset is composed of multiple tables covering players, teams, performances, market values, and transfers.  
Together, they allow us to analyze player careers and team dynamics from several perspectives.

The available dataframes are:

- **player_injured_df** → Records of player injuries, including dates and types.  
- **player_latest_market_value_df** → Latest known market value of each player.    
- **player_performances_df** → Club-level performances such as matches, goals, assists.    
- **team_competitions_seasons_df** → Teams’ participation in competitions by season.  
- **team_details_df** → General metadata about teams (name, country, founding year, etc.).  


## 📂 Data Storage
The raw data is **not stored in this repository** (to keep it lightweight).  
Instead, all files live in **Google Drive**.

To ensure reproducibility, we provide a script (`download_data.py`) that downloads all necessary folders automatically.


## 📥 How to Acces the Data
1. Install requirements:
you open a new terminal and write: pip install gdown
2. run the (`download_data.py`) file 

## 📂 Estructura del Proyecto

- **`_reports/`** → Contiene reportes estadísticos antes y después de limpiar los datos (media, mediana, valores nulos, etc.).
- **`plots/`** → Contiene visualizaciones (como histogramas, barras, etc.) que representan la distribución de las variables más relevantes.
- **`Main_code.py`** → Script principal que realiza la limpieza de los datos, genera los reportes estadísticos y crea las visualizaciones.
- **`download_data.py`** → Script para descargar automáticamente los archivos de datos desde una fuente externa.
- **`.gitignore`** → Evita que archivos innecesarios (como temporales, entornos virtuales o datos sensibles) se suban al repositorio.