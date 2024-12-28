# -*- coding: utf-8 -*-
"""medusaVariant.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1JQc9vLMzOoTH3_Iqj1sSe9xJ7pkFmHYh
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torchinfo import summary
from datasets import load_dataset, Dataset
from torch.optim import Adam
import torch.nn.functional as F
import os
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


device = "mps"
torch.set_default_device(device)
model_name = "gpt2"
model_dir = "./gpt2model"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token
print(summary(model))


local_processed_dataset_path = "./self_distilled_wikitext"
# Load or download and process the dataset
if not os.path.exists(local_processed_dataset_path):
    wiki_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    wiki_data = wiki_data.select(range(1000))

    # Preprocess data
    def preprocess(data):
        data = data["text"]
        tokens = tokenizer(
            data,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=50,
            truncation=True,
            padding="longest",
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        outputs = model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_length=100,
            do_sample=True,
            temperature=0.9,
        )
        outputs = outputs[0]
        return {"input_ids": outputs[:-2], "labels": [outputs[-1]]}

    def filter_test(data):
        tokens = tokenizer(
            data["text"], add_special_tokens=False, truncation=True, max_length=50
        )
        return len(tokens["input_ids"]) > 20

    dataset = wiki_data.filter(filter_test)
    tokenized_two_adv = dataset.map(preprocess)
    tokenized_two_adv.save_to_disk(local_processed_dataset_path)
else:
    tokenized_two_adv = Dataset.load_from_disk(local_processed_dataset_path)

dataloader = DataLoader(
    tokenized_two_adv,
    generator=torch.Generator(device="mps"),
    batch_size=16,
    shuffle=True,
    collate_fn=lambda batch: {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item["input_ids"]) for item in batch], batch_first=True
        ),
        "labels": torch.tensor([item["labels"] for item in batch]),
    },
)


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
)

two_adv_model = get_peft_model(model, lora_config)


epochs = 50
lr = 3e-4
batch_size = 16
two_adv_model = two_adv_model.to(device)
optimizer = Adam(two_adv_model.parameters(), lr=lr)


def loss_calc(model, input_ids, labels):
    outputs = model(input_ids=input_ids)
    # print(outputs.logits.shape)  # (batch_dim,tok_pos,logit)
    logits = outputs.logits[:, -1, :]
    loss = F.cross_entropy(logits, labels.view(-1))
    return loss


for epoch in range(epochs):

    two_adv_model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        # print(input_ids.shape)
        # labels = input_ids[:, 1:].contiguous()
        batch_num = batch["labels"].size(0)
        labels = batch["labels"].reshape((batch_num,)).to(device)
        optimizer.zero_grad()
        loss = loss_calc(two_adv_model, input_ids, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

model_path = "./medusa_adjusted_model"

two_adv_model.save_pretrained(model_path)