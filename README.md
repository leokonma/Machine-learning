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
you open a new terminal and write: pip install -r requirements.txt
2. run the (`download_data.py`) file 
3. run the (`Data_module.py`) file 
4. you are able now to run Model_Module.ipynb choosing the kernel (.venv)

## 📂 Estructura del Proyecto

- **`_reports/`** → Contiene reportes estadísticos antes y después de limpiar los datos (media, mediana, valores nulos, etc.), asi como el codigo optimizado por chat gpt de la primera entrega del proyecto con los codigos que aplican el reporte.
- **`plots/`** → Contiene visualizaciones (como histogramas, barras, etc.) que representan la distribución de las variables más relevantes, pertenceientes de las variables basicas de la primera entrega.
- **`Data_Module.py`** → Script principal que realiza la limpieza y enriquecimiento de los datos, asi como preparaciones enfocadas a la funcion objetivo   .
- **`Model_module.ipynb`** → Script para desarrollar algoritmos de ML.
- **`.gitignore`** → Evita que archivos innecesarios (como temporales, entornos virtuales o datos sensibles) se suban al repositorio.