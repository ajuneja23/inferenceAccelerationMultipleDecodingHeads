from tokenPPO import calculateTrajectories
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os


model_path = "./gpt2model"
traj_path = "./traj_data"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

results = []

if not os.path.exists(traj_path):
    wiki_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    raw_data = []
    good_count = 0
    i = 0
    while good_count < 450:
        if len(wiki_data[i]["text"]) >= 1000:
            good_count += 1
            raw_data.append(wiki_data[i])
        i += 1
    wiki_data = raw_data
    count = 0
    for data in wiki_data:
        count += 1
        prompt = data["text"]
        results.append(calculateTrajectories(model, tokenizer, prompt))
        print(f"{count}/1000 done")
    torch.save(results, traj_path)
