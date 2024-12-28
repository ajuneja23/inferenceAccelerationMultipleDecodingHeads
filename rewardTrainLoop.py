import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenPPO import RewardModel, reward_training_loop
import torch.nn as nn
from torchinfo import summary

model_path = "./gpt2model"


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

traj_data = torch.load("traj_data")

a, b, c, d = traj_data[0]

print(len(d))
print(d[0].shape)
print(type(d))


base_reward = RewardModel()
print(summary(base_reward))


def trainingReward(epochs):
    for i in range(epochs):
        reward_training_loop(base_reward, model, traj_data)
        print(f"Done with Epoch {i+1}/{epochs}")
        with open("reward_loss.txt", "a") as file:
            file.write(f"Done with Epoch {i+1}/{epochs}\n")


epochs = 75
trainingReward(epochs)


torch.save(base_reward, "trained_reward_model.pth")
