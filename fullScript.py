from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType


device = "mps"
steps = 20
reward_model_path = "trained_reward_model.pth"
base_gpt_path = "./gpt2model"
reward_model = torch.load(reward_model_path)
reward_model = reward_model.to(device)
base_gpt = AutoModelForCausalLM.from_pretrained(base_gpt_path)
base_gpt = base_gpt.to(device)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
train_gpt = AutoModelForCausalLM.from_pretrained(base_gpt_path)
train_gpt = get_peft_model(train_gpt, peft_config)
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


def calculateKL(model1, model2, prompt, tokenizer):
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=50,
        truncation=True,
        padding="longest",
    )
    input_ids = tokens["input_ids"].to(device)
    output1 = model1(input_ids=input_ids)
    output2 = model2(input_ids=input_ids)
    logits1 = output1.logits[0, -1, :].squeeze()
    logits1 = torch.softmax(logits1, dim=0)
    logits2 = output2.logits[0, -1, :].squeeze()
    logits2 = torch.softmax(logits2, dim=0)
    kl = torch.nn.functional.kl_div(logits2.log(), logits1, reduction="sum")
    return kl


def calculateTrajectories(model, tokenizer, prompt):
    tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=50,
        truncation=True,
        padding="longest",
    )
    input_ids = tokens["input_ids"]
    input_ids = input_ids.to(device)
    full_logits = []
    final_token_hidden_states = []
    model = model.to(device)
    with torch.no_grad():
        for i in range(steps):
            output = model(input_ids=input_ids, output_hidden_states=True)
            logits = output.logits[0, -1, :].squeeze()
            logits = torch.softmax(logits, dim=0).tolist()
            next_token = torch.argmax(output.logits[:, -1, :]).item()
            full_logits.append(logits)
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_token]]).to(device)), dim=-1
            )
            input_ids = input_ids.to(device)
            final_token_hidden_states.append(output.hidden_states[-1][0, -1, :])
            del output
    return tokens["input_ids"], input_ids, full_logits, final_token_hidden_states


def train_llm(
    reward_model, base_gpt, train_gpt, tokenizer, prompts, beta=0.25, epochs=10, lr=3e-4
):
    optimizer = torch.optim.Adam(train_gpt.parameters(), lr=lr)
    train_gpt.train()
    embedding_layer = base_gpt.transformer.wte
    for epoch in range(epochs):
        count = 0
        for prompt in prompts:
            count += 1
            loss = 0
            prompt = prompt["prompt"]
            (start_toks, full_toks, full_logits, final_hidden_states) = (
                calculateTrajectories(train_gpt, tokenizer, prompt)
            )
            start_index = start_toks.shape[1]
            for i in range(start_index, full_toks.shape[1]):
                output = train_gpt(input_ids=full_toks[0, : i + 1])
                predicted_token_id = torch.argmax(
                    output.logits[-1].squeeze(), dim=0
                ).item()
                token_embedding = embedding_layer(
                    torch.tensor([predicted_token_id]).to(device)
                ).squeeze()
                reward = reward_model(
                    final_hidden_states[i - start_index], token_embedding
                )
                iter_loss = reward - beta * calculateKL(
                    base_gpt, train_gpt, prompt, tokenizer
                )
                iter_loss *= -1
                loss += iter_loss
            print(
                f"Epoch {epoch+1}/{epochs} Batch {count}/{len(prompts)}: {loss.item()}"
            )
            with open("train_gpt_loss.txt", "a") as file:
                file.write(
                    f"Epoch {epoch+1}/{epochs} Batch {count}/{len(prompts)}: {loss.item()}"
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{epochs} completed")
        with open("train_gpt_loss.txt", "a") as file:
            file.write(f"Epoch {epoch+1}/{epochs} completed")


train_llm(reward_model, base_gpt, train_gpt, tokenizer, prompts)
torch.save(train_gpt, "train_gpt_post.pth")
