import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from constants import DATA_DIR, MODEL_DIR, RESULTS_DIR, HIDDEN_DIM, NUM_HEADS, LEARNING_RATE, NUM_EPOCHS, ANOMALY_THRESHOLD
from preprocess import build_graph, introduce_anomalies, split_data
from model import GraphTransformerAutoencoder, GraphTransformerVAE

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_model(model, train_loader, val_loader=None, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device="cpu"):
    """
    Train the model with early stopping based on validation loss
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            # Handle different model types
            if isinstance(model, GraphTransformerVAE):
                x_hat, mu, logvar = model(data)
                loss, recon_loss, kl_loss = model.loss_function(data.x, x_hat, mu, logvar)
                if epoch % 10 == 0 and len(train_loader) < 5:  # Only for small datasets
                    print(f"    Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}")
            else:
                x_hat, _ = model(data)
                loss = ((data.x - x_hat)**2).mean()
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation if provided
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    if isinstance(model, GraphTransformerVAE):
                        x_hat, mu, logvar = model(data)
                        loss, _, _ = model.loss_function(data.x, x_hat, mu, logvar)
                    else:
                        x_hat, _ = model(data)
                        loss = ((data.x - x_hat)**2).mean()
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            val_msg = f", Val Loss: {avg_val_loss:.4f}" if val_loader else ""
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_train_loss:.4f}{val_msg}")
    
    # Load best model if validation was used
    if val_loader and os.path.exists(os.path.join(MODEL_DIR, 'best_model.pt')):
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'final_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curve.png'))
    
    return model

def evaluate_anomaly_detection(model, test_data, anomaly_labels=None, threshold=None, device="cpu"):
    """
    Evaluate the model's anomaly detection performance
    If anomaly_labels are provided, calculate ROC and PR curves
    """
    model.eval()
    test_data = test_data.to(device)
    
    # Get anomaly scores
    with torch.no_grad():
        if isinstance(model, GraphTransformerVAE):
            x_hat, mu, logvar = model(test_data)
            recon_error = model.reconstruction_error(test_data.x, x_hat)
            kl_div = model.kl_divergence(mu, logvar)
            anomaly_scores = recon_error + 0.1 * kl_div
        else:
            x_hat, _ = model(test_data)
            anomaly_scores = model.reconstruction_error(test_data.x, x_hat)
    
    # Convert to numpy for analysis
    scores = anomaly_scores.cpu().numpy()
    
    # If threshold is None, use percentile
    if threshold is None:
        threshold = np.percentile(scores, ANOMALY_THRESHOLD * 100)
    
    # Detect anomalies
    predicted_anomalies = (scores > threshold).astype(int)
    
    # If we have true labels, calculate metrics
    results = {
        'anomaly_scores': scores,
        'threshold': threshold,
        'predicted_anomalies': predicted_anomalies
    }
    
    if anomaly_labels is not None:
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(anomaly_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(anomaly_labels, scores)
        pr_auc = auc(recall, precision)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Plot PR curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'roc_pr_curves.png'))
        
        # Calculate additional metrics
        tp = ((predicted_anomalies == 1) & (anomaly_labels == 1)).sum()
        fp = ((predicted_anomalies == 1) & (anomaly_labels == 0)).sum()
        tn = ((predicted_anomalies == 0) & (anomaly_labels == 0)).sum()
        fn = ((predicted_anomalies == 0) & (anomaly_labels == 1)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        results.update({
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1_score,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    
    return results

def analyze_anomalies(model, data, node_ids, results):
    """
    Analyze top anomalies and their features
    """
    anomaly_scores = results['anomaly_scores']
    threshold = results['threshold']
    
    # Get indices of anomalies
    anomaly_indices = np.where(anomaly_scores > threshold)[0]
    
    # Create a DataFrame with anomalies
    anomaly_data = {
        'node_id': [node_ids[i] if i < len(node_ids) else f"Unknown-{i}" for i in anomaly_indices],
        'anomaly_score': anomaly_scores[anomaly_indices],
    }
    
    # Add top feature contributions if available
    if hasattr(data, 'x') and data.x is not None:
        x = data.x.cpu().numpy() if torch.is_tensor(data.x) else data.x
        # Get reconstructed data
        with torch.no_grad():
            if isinstance(model, GraphTransformerVAE):
                x_hat, _, _ = model(data)
            else:
                x_hat, _ = model(data)
        x_hat = x_hat.cpu().numpy()
        
        # Calculate feature-wise error
        feature_errors = (x - x_hat)**2
        
        # For each anomaly, identify top contributing features
        for i, idx in enumerate(anomaly_indices):
            if idx < len(feature_errors):
                # Get top 3 features with highest error
                top_features = np.argsort(feature_errors[idx])[-3:][::-1]
                for j, feat_idx in enumerate(top_features):
                    anomaly_data[f'top_feature_{j+1}'] = feat_idx
                    anomaly_data[f'feature_{j+1}_error'] = feature_errors[idx][feat_idx]
    
    anomaly_df = pd.DataFrame(anomaly_data)
    anomaly_df.to_csv(os.path.join(RESULTS_DIR, 'anomalies.csv'), index=False)
    
    print(f"\nDetected {len(anomaly_indices)} anomalies out of {len(anomaly_scores)} nodes")
    print(f"Top 10 anomalies:")
    if len(anomaly_df) > 0:
        print(anomaly_df.sort_values('anomaly_score', ascending=False).head(10))
    
    return anomaly_df

def main():
    print("Loading and preprocessing data...")
    graph = build_graph(DATA_DIR)
    
    if graph.x is None or graph.x.shape[0] == 0:
        print("Error: No data found or empty graph. Please check the data directory.")
        return
    
    print(f"Graph loaded with {graph.x.shape[0]} nodes and {graph.edge_index.shape[1]} edges")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Split data for training and testing
    train_data, test_data = split_data(graph, test_size=0.2)
    
    # Introduce synthetic anomalies to test data for evaluation
    test_data, anomaly_labels = introduce_anomalies(test_data, anomaly_ratio=0.05)
    
    # Create data loaders
    train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)
    
    # Initialize the model (choose between standard AE and VAE)
    in_channels = graph.x.shape[1]
    use_vae = True  # Toggle between standard AE and VAE
    
    print(f"\nInitializing Graph Transformer {'VAE' if use_vae else 'Autoencoder'}")
    print(f"Input dimensions: {in_channels}, Hidden dimensions: {HIDDEN_DIM}, Heads: {NUM_HEADS}")
    
    if use_vae:
        model = GraphTransformerVAE(
            in_channels=in_channels, 
            hidden_channels=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            latent_dim=HIDDEN_DIM * 2,
            dropout=0.1
        )
    else:
        model = GraphTransformerAutoencoder(
            in_channels=in_channels, 
            hidden_channels=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            dropout=0.1
        )
    
    # Train the model
    print("\nTraining model...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )
    
    # Evaluate anomaly detection
    print("\nEvaluating anomaly detection...")
    results = evaluate_anomaly_detection(
        model=model,
        test_data=test_data,
        anomaly_labels=anomaly_labels,
        device=device
    )
    
    # Print metrics
    if 'roc_auc' in results:
        print(f"\nEvaluation Metrics:")
        print(f"ROC AUC: {results['roc_auc']:.4f}")
        print(f"PR AUC: {results['pr_auc']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
    
    # Analyze anomalies
    print("\nAnalyzing detected anomalies...")
    node_ids = [i for i in range(test_data.x.shape[0])]  # Replace with actual node IDs if available
    anomaly_df = analyze_anomalies(model, test_data, node_ids, results)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    return results

if __name__ == "__main__":
    main()
