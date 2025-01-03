import matplotlib.pyplot as plt
import re

# Read the file
with open("train_gpt_loss.txt", "r") as file:
    lines = file.readlines()

batch_losses = []
epoch_avg_losses = []
current_epoch_losses = []

for line in lines:
    line = line.strip()

    # Match batch loss line
    batch_match = re.match(r"Epoch \d+/\d+ Batch \d+/\d+: ([\d.]+)", line)
    if batch_match:
        loss = float(batch_match.group(1))
        batch_losses.append(loss)
        current_epoch_losses.append(loss)

    # Match epoch completion line
    epoch_match = re.match(r"Epoch \d+/\d+ completed", line)
    if epoch_match and current_epoch_losses:
        epoch_avg = sum(current_epoch_losses) / len(current_epoch_losses)
        epoch_avg_losses.append(epoch_avg)
        current_epoch_losses = []

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot per-batch losses
ax1.plot(batch_losses)
ax1.set_title("Loss per Batch")
ax1.set_xlabel("Batch Number")
ax1.set_ylabel("Loss")
ax1.grid(True)

# Plot average loss per epoch
ax2.plot(epoch_avg_losses, "r-")
ax2.set_title("Average Loss per Epoch")
ax2.set_xlabel("Epoch Number")
ax2.set_ylabel("Average Loss")
ax2.grid(True)

plt.tight_layout()
plt.show()
