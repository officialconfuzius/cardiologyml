import torch
from torch_geometric.nn import GraphUNet
from torch_geometric.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pickle
from model.FaceGraphUNetModel import *
from model.MeshDataset import CustomMeshDataset

if __name__ == "__main__":
    # Set some parameters
    batch_size = 4
    learning_rate = 0.001
    num_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data from the directory
    dataset = CustomMeshDataset()
    
    # Instantiate the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = FaceGraphUNetModel(in_channels=3, hidden_channels=32, out_channels=12, depth=3)  # 3D centroid input
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Track the training loss for each epoch
    epoch_losses = []
    loss_file = 'epoch_losses.pkl'

    # Check if the loss file already exists
    if os.path.exists(loss_file):
        with open(loss_file, 'rb') as f:
            epoch_losses = pickle.load(f)
    else:
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                # Move data to the appropriate device
                vertices = batch['x'].to(device)  # (N, 3)
                edge_index = batch['edge_index'].to(device)  # (M, K)
                labels = batch['y'].to(device)  # (N,)
                
                # Squeeze due to added batch dimension
                if edge_index.ndim == 3: 
                    vertices = vertices.squeeze(0)
                    edge_index = edge_index.squeeze(0)
                    labels = labels.squeeze(0)
                
                # Forward pass
                outputs = model(vertices, edge_index)  # Output should be of shape (N, num_classes)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item()
            
            # Average loss for the epoch
            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save the model
        torch.save(model.state_dict(), 'trained_model.pth')
        print("Model saved to 'trained_model.pth'")

        # Save the losses to a file
        with open(loss_file, 'wb') as f:
            pickle.dump(epoch_losses, f)

    # Plot the training loss over epochs
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.show()
