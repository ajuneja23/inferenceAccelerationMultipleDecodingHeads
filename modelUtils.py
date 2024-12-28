from torchinfo import summary
import torch
from transformers import AutoModelForCausalLM


model_path = "./gpt2model"
model = AutoModelForCausalLM.from_pretrained(model_path)
print(summary(model))

print(model.transformer.h[0])
