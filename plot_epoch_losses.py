import os
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Determine the path to the saved loss file
script_path = Path(__file__).resolve().parent  # Directory where the script is located
loss_file = script_path / 'epoch_losses.pkl'  # Use this directory to define the path to the loss file
train_file = script_path / 'train.py'  # Path to the train.py file

# Extract hyperparameters from train.py
if train_file.exists():
    with open(train_file, 'r') as f:
        train_content = f.read()

    # Use regex to extract hyperparameter values
    batch_size = re.search(r"batch_size\s*=\s*(\d+)", train_content)
    learning_rate = re.search(r"learning_rate\s*=\s*([\d.]+)", train_content)
    num_epochs = re.search(r"num_epochs\s*=\s*(\d+)", train_content)
    hidden_channels = re.search(r"hidden_channels\s*=\s*(\d+)", train_content)
    depth = re.search(r"depth\s*=\s*(\d+)", train_content)

    # Assign extracted values or default to 'Unknown'
    batch_size = batch_size.group(1) if batch_size else 'Unknown'
    learning_rate = learning_rate.group(1) if learning_rate else 'Unknown'
    num_epochs = num_epochs.group(1) if num_epochs else 'Unknown'
    hidden_channels = hidden_channels.group(1) if hidden_channels else 'Unknown'
    depth = depth.group(1) if depth else 'Unknown'

else:
    print(f"train.py file not found: {train_file}")
    batch_size = learning_rate = num_epochs = hidden_channels = depth = 'Unknown'

# Load the losses
if loss_file.exists():
    with open(loss_file, 'rb') as f:
        epoch_losses = pickle.load(f)

    # Plot the training loss over epochs
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)

    # Add hyperparameter info to the plot
    hyperparams_text = (
        f"Batch Size: {batch_size}\n"
        f"Learning Rate: {learning_rate}\n"
        f"Num Epochs: {num_epochs}\n"
        f"Hidden Channels: {hidden_channels}\n"
        f"Depth: {depth}"
    )
    plt.gcf().text(0.7, 0.98, hyperparams_text, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Show the plot
    plt.show()
else:
    print(f"Loss file not found: {loss_file}")
