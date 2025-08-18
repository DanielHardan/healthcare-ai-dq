from constants import DATA_DIR
from download_data import download_and_extract
from preprocess import build_graph
from detect_anomalies import main as detect_main

def main():
	print("Step 1: Downloading Synthea data...")
	download_and_extract()
	print("Step 2: Preprocessing data...")
	build_graph(DATA_DIR)
	print("Step 3: Running anomaly detection...")
	detect_main()

if __name__ == "__main__":
	main()
