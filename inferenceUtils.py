import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor


model1_dir = "./gpt2model"
model2_dir = "./medusa_adjusted_model"
model1 = AutoModelForCausalLM.from_pretrained(model1_dir, torch_dtype="auto")
model2 = AutoModelForCausalLM.from_pretrained(model2_dir, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model1_dir)
tokenizer.pad_token = tokenizer.eos_token
device = "mps"
model1 = model1.to(device)
model2 = model2.to(device)


def generate_tokens_parallel(tokens, attention_mask):
    with ThreadPoolExecutor() as executor:
        future_model1 = executor.submit(
            genModel1OneToken, {"input_ids": tokens, "attention_mask": attention_mask}
        )
        future_model2 = executor.submit(
            genModel2OneToken, {"input_ids": tokens, "attention_mask": attention_mask}
        )

        new_token_model1 = future_model1.result()
        new_token_model2 = future_model2.result()
    print(tokens.shape)
    print(torch.tensor([[new_token_model1, new_token_model2]]).shape)
    tokens = torch.cat(
        [tokens, torch.tensor([[new_token_model1, new_token_model2]], device="mps")],
        dim=1,
    )
    print(tokens.shape)
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones(1, 1, device=attention_mask.device),
            torch.ones(1, 1, device=attention_mask.device),
        ],
        dim=-1,
    )
    return tokens, attention_mask


def generate_from_tokenized(
    tokens, attention_mask, eos_token_id=50256, max_new_tokens=100
):
    num_toks = 0
    while num_toks < max_new_tokens:
        tokens, attention_mask = generate_tokens_parallel(tokens, attention_mask)
        num_toks += 2
        if tokens[0][-2] == eos_token_id or num_toks > max_new_tokens:
            num_toks -= 1
            tokens[0] = tokens[0][:-1]
            break
    return tokens


def genModel1OneToken(tokenized_input):
    output = model1.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        max_new_tokens=1,
        do_sample=True,
        temperature=0.9,
    )
    print(tokenized_input["input_ids"].shape)
    print(tokenized_input["input_ids"])
    print(tokenized_input["attention_mask"].shape)
    print(tokenized_input["attention_mask"])
    generated_token_id = output[0, -1].item()
    return generated_token_id


def genModel2OneToken(tokenized_input):
    output = model2.generate(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        max_new_tokens=1,
        do_sample=True,
        temperature=0.9,
    )
    generated_token_id = output[0, -1].item()
    return generated_token_id


def generate(text):
    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=100,
        truncation=True,
        padding="longest",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    tokens = generate_from_tokenized(tokens["input_ids"], tokens["attention_mask"])
    print(tokens)
    generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return generated_text


print(generate("How are you doing today?"))
