import os
import requests
import zipfile
from constants import DATA_DIR, SYNTHEA_URL

os.makedirs(DATA_DIR, exist_ok=True)
ZIP_PATH = os.path.join(DATA_DIR, "synthea.zip")

def download_and_extract():
    # Check if data is already extracted (look for a known file/folder)
    if os.path.exists(ZIP_PATH):
        print("Data already downloaded and extracted. Skipping download.")
        return
    print("Downloading Synthea FHIR data...")
    r = requests.get(SYNTHEA_URL, stream=True)
    with open(ZIP_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print("Done. FHIR data is ready in the data directory.")

if __name__ == "__main__":
    download_and_extract()
