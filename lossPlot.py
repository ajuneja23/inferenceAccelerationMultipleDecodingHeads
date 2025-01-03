import matplotlib.pyplot as plt

# Step 1: Read and Parse the File
rewards_per_batch = []
avg_rewards_per_epoch = []
with open("train_gpt_reward.txt", "r") as file:
    current_epoch_rewards = []
    for line in file:
        line = line.strip()
        if line.startswith("Epoch") and "Batch" in line:
            # Extract reward for the batch
            reward = float(line.split(":")[-1].strip())
            rewards_per_batch.append(reward)
            current_epoch_rewards.append(reward)
        elif line.startswith("Epoch") and "completed" in line:
            # Calculate average reward for the epoch
            avg_reward = sum(current_epoch_rewards) / len(current_epoch_rewards)
            avg_rewards_per_epoch.append(avg_reward)
            current_epoch_rewards = []

# Step 2: Plot Batch-by-Batch Reward Growth
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(rewards_per_batch, label="Batch Reward")
plt.title("Batch-by-Batch Reward Growth")
plt.xlabel("Batch Number")
plt.ylabel("Reward")
plt.legend()

# Step 3: Plot Average Reward per Epoch
plt.subplot(1, 2, 2)
plt.plot(
    range(1, len(avg_rewards_per_epoch) + 1),
    avg_rewards_per_epoch,
    marker="o",
    label="Avg Reward per Epoch",
)
plt.title("Average Reward per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.legend()

# Show the plots
plt.tight_layout()
plt.savefig("train_gpt_reward_plot.png")
