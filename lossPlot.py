import matplotlib.pyplot as plt

# File path
file_path = "loss.txt"

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
