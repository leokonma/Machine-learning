# src/download_data.py
from pathlib import Path
import gdown

# Map your Drive folder IDs here (name: folder_id)
FOLDERS = {
    # EXAMPLES — replace with your real IDs
    "player_injuries": "1ZDfFfdR2ZOBNuHov2g6nbOWh0QdEXhRX",
    "player_latest_market_value": "16KeVV_KpdOiHjgNrGKROtlXrQuI5AhRg",
    "player_market_value": "17HJHRDVIqGhARlCqxkLag68QR3thT1Oh",
    "player_national_performances": "1Pmwbpsen8UWsOYbac7wAjt7TlBUSX-kC",
    "player_performances": "1o_mmxALx-JTPmjGbRTv0oixtff7uQzSR",
    "player_profiles": "1Otah8KtCGIR-5xeSBbMI6xBn3gt8q_La",
    "player_teammates_played_with": "1FTjADdSdEWi-5EHJImQEsfmn_Wnxa8ys",
    "team_children": "1OJ5G0AeDTD3G7RnzBA3ybOcUD1PKjOUC",
    "team_competitions_seasons": "1W29kM_rvS2IWAw6FXzlFzDFRLt_zSNjR",
    "team_details": "1NI1A__ZNfd-dFDXeTWJzX3GO0B9H_McZ",
    "transfer_history": "1UysW0OF00PuCDoHhY-OIp6lXwsOzVV-X",
}
 
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def download_folder(name: str, folder_id: str):
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=folder_id, output=str(out), quiet=False, use_cookies=False)

def main():
    for name, fid in FOLDERS.items():
        print(f"\n=== Downloading {name} ===")
        download_folder(name, fid)
    print("\nAll folders downloaded to data/raw/.")

if __name__ == "__main__":
    main()
