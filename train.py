from model.FaceGraphUNetModel import *
from model.MeshObject import *
from model.MeshDataset import CustomMeshDataset
from torch.utils.data import DataLoader
from pathlib import Path
import pickle
import neptune
from neptune.utils import stringify_unsupported
import os
from neptune_pytorch import NeptuneLogger

if __name__ == "__main__":
    # Set up neptune logging
    nep_token = os.getenv("NEPTUNE_API_TOKEN")
    run = neptune.init_run(api_token=nep_token, project="cardiologyml/Cardiology-ml")
    
    # Set some parameters
    batch_size = 4
    learning_rate = 0.01
    num_epochs = 10
    in_channels = 3
    out_channels = 12
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the output directory
    script_path = Path(__file__).resolve().parent
    model_file = script_path / 'trained_model.pth'
    loss_file = script_path / 'epoch_losses.pkl'

    # Load the data from the directory
    dataset = CustomMeshDataset()
    
    # Instantiate the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = FaceGraphUNetModel(in_channels=in_channels, hidden_channels=32, out_channels=out_channels, depth=3)  # 3D centroid input
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Logging params
    log_params = {
        "lr": learning_rate,
        "bs": batch_size,
        "in_channels": in_channels, 
        "out_channels": out_channels,
        "model_filename": "basemodel",
        "device": device,
        "epochs": num_epochs
    }
    
    # Set up Neptune logging instance
    logger = NeptuneLogger(
    run=run,
    model=model,
    log_model_diagram=True,
    log_gradients=True,
    log_parameters=True,
    log_freq=30
    )

    run[logger.base_namespace]["hyperparams"] = stringify_unsupported( 
        log_params
    )
    
    # Training loop
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(dataloader, 0):
            # Move data to the appropriate device
            vertices = data['x'].to(device)  # (N, 3)
            edge_index = data['edge_index'].to(device)  # (M, K)
            labels = data['y'].to(device)  # (N,)
            
            # Squeeze due to added batch dimension
            if edge_index.ndim == 3: 
                vertices = vertices.squeeze(0)
                edge_index = edge_index.squeeze(0)
                labels = labels.squeeze(0)
            
            # Forward pass
            outputs = model(vertices, edge_index)  # Output should be of shape (N, num_classes)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Log loss every 30 steps
            if i % 30 == 0: 
                run[logger.base_namespace]["batch/loss"].append(loss.item())
            
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
        
        # Log checkpoint using neptune
        logger.log_checkpoint()

    # Save the model
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to '{model_file}'")

    # Save the epoch losses to a file
    with open(loss_file, 'wb') as f:
        pickle.dump(epoch_losses, f)
    print(f"Losses saved to '{loss_file}'")

    # Sync data to neptune
    run.stop()