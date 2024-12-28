from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tokenPPO import train_llm
from datasets import load_dataset


device = "mps"

reward_model_path = "trained_reward_model.pth"
base_gpt_path = "./gpt2model"
reward_model = torch.load(reward_model_path)
reward_model = reward_model.to(device)
base_gpt = AutoModelForCausalLM.from_pretrained(base_gpt_path)
base_gpt = base_gpt.to(device)
train_gpt = AutoModelForCausalLM.from_pretrained(base_gpt_path)
train_gpt = train_gpt.to(device)
tokenizer = AutoTokenizer.from_pretrained(base_gpt_path)

wiki_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
i = 0
good_count = 0
prompts = []
while good_count < 900:
    if len(wiki_data[i]["text"]) >= 1000:
        good_count += 1
        if good_count >= 450:
            prompts.append({"prompt": wiki_data[i]["text"]})
    i += 1


train_llm(reward_model, base_gpt, train_gpt, tokenizer, prompts)
