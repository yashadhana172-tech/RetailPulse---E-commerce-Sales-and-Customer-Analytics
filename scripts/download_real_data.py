"""
RetailPulse - Download Real UCI Online Retail Dataset Script

Official Target:
- Dataset: Online Retail Data Set (UCI Machine Learning Repository)
- Donor: Dr Daqing Chen, London South Bank University
- DOI / URL: https://archive.ics.uci.edu/dataset/352/online+retail
- Direct Zip: https://archive.ics.uci.edu/static/public/352/online+retail.zip
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Instructions:
Run: python scripts/download_real_data.py
The script will download the official zip archive from UCI, extract the Excel
spreadsheet 'Online Retail.xlsx', and place it in 'data/raw/Online_Retail.xlsx'.
"""

import os
import sys
import zipfile
import requests

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

def download_uci_dataset(dest_dir: str = "data/raw") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "online_retail.zip")
    extracted_target = os.path.join(dest_dir, "Online Retail.xlsx")

    if os.path.exists(extracted_target) and os.path.getsize(extracted_target) > 1000000:
        print(f"Dataset already downloaded and extracted at: {extracted_target}")
        return extracted_target

    print(f"Downloading official UCI Online Retail dataset from {UCI_ZIP_URL}...")
    headers = {"User-Agent": "RetailPulse-Analytics-Portfolio/1.0"}
    response = requests.get(UCI_ZIP_URL, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"Downloaded zip file to {zip_path} ({os.path.getsize(zip_path)} bytes). Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_dir)

    print(f"Successfully extracted dataset to {dest_dir}!")
    return extracted_target

if __name__ == "__main__":
    try:
        path = download_uci_dataset()
        print(f"Ready: {path}")
    except Exception as e:
        print(f"Download failed or network unavailable: {e}")
        print("Note: RetailPulse operates fully offline with deterministic synthetic demo data in data/sample/.")
        sys.exit(1)
