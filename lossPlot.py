import matplotlib.pyplot as plt

# File path
"""file_path = "loss.txt"

# Initialize lists
epochs = []
losses = []

# Read the file and extract data
with open(file_path, "r") as file:
    for line in file:
        # Example line format: "Epoch 50/50, Loss: 0.1917"
        parts = line.strip().split(", ")
        epoch_info = parts[0].split(" ")[1]  # Extract the current epoch
        loss_value = float(parts[1].split(": ")[1])  # Extract the loss value

        epochs.append(int(epoch_info.split("/")[0]))  # Add epoch number
        losses.append(loss_value)  # Add loss value

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, marker="o", label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save or show the plot
plt.savefig("loss_plot.png")  # Save the plot as a PNG file
plt.show()  # Display the plot
"""
import matplotlib.pyplot as plt

# Filepath to the reward loss file
file_path = "reward_loss.txt"

# Initialize lists to store batch losses and epoch averages
batch_losses = []
epoch_averages = []
current_epoch_losses = []

# Read and process the file
with open(file_path, "r") as file:
    for line in file:
        if line.startswith("Batch Loss:"):
            # Extract batch loss value
            loss_value = float(line.split("Batch Loss:")[1].strip())
            batch_losses.append(loss_value)
            current_epoch_losses.append(loss_value)
        elif line.startswith("Done with Epoch"):
            # Calculate and store the average loss for the epoch
            if current_epoch_losses:
                epoch_average = sum(current_epoch_losses) / len(current_epoch_losses)
                epoch_averages.append(epoch_average)
                current_epoch_losses = []  # Reset for the next epoch

# Plotting
plt.figure(figsize=(12, 6))

# Subplot 1: Batch loss over time
plt.subplot(1, 2, 1)
plt.plot(batch_losses, label="Batch Loss", color="blue", linewidth=1)
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.title("Batch Loss Over Time")
plt.legend()

# Subplot 2: Average loss per epoch
plt.subplot(1, 2, 2)
plt.plot(
    range(1, len(epoch_averages) + 1),
    epoch_averages,
    marker="o",
    color="green",
    label="Average Loss per Epoch",
)
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Average Loss per Epoch")
plt.legend()

# Show the plot
plt.tight_layout()
# Save the plot to a file
plt.savefig("reward_model_loss.png")
