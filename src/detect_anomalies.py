import torch
from constants import DATA_DIR
from preprocess import build_graph
from model import GraphTransformerAutoencoder

def main():
    graph = build_graph(DATA_DIR)
    model = GraphTransformerAutoencoder(graph.x.shape[1], 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = torch.tensor(graph.x, dtype=torch.float)
    edge_index = graph.edge_index
    data = graph
    # Training loop
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        x_hat, _ = model(data)
        loss = ((x - x_hat)**2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    # Anomaly scores
    model.eval()
    x_hat, _ = model(data)
    scores = model.reconstruction_error(x, x_hat)
    print("Top anomalies:")
    print(scores.topk(10))

if __name__ == "__main__":
    main()
