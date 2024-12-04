import pickle
import matplotlib.pyplot as plt

# Path to the saved loss file
loss_file = 'epoch_losses.pkl'  # Adjust the path if necessary, e.g., 'output/epoch_losses.pkl'

# Load the losses
with open(loss_file, 'rb') as f:
    epoch_losses = pickle.load(f)

# Plot the training loss over epochs
plt.figure()
plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.grid(True)
plt.show()
