import os
import torch
from constants import DATA_DIR, MODEL_DIR, RESULTS_DIR
from download_data import download_and_extract
from preprocess import build_graph
from detect_anomalies import main as detect_main

def main():
    # Create necessary directories
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("Healthcare Data Quality Assessment using Graph Transformer Autoencoder")
    print("-" * 70)
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nStep 1: Downloading Synthea data...")
    download_and_extract()
    
    print("\nStep 2: Preprocessing data and building graph...")
    graph = build_graph(DATA_DIR)
    print(f"Created graph with {graph.x.shape[0]} nodes and {graph.edge_index.shape[1]} edges")
    print(f"Each node has {graph.x.shape[1]} features")
    
    print("\nStep 3: Running anomaly detection pipeline...")
    results = detect_main()
    
    print("\nProcess complete!")
    print(f"Results are saved in the {RESULTS_DIR} directory")

if __name__ == "__main__":
    main()
