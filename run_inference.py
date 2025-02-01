from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "cpu"

from concurrent.futures import ThreadPoolExecutor

# Load the trained model
model_path = "final_final_trained_model.pth"
model1 = AutoModelForCausalLM.from_pretrained("./gpt2model")
model2 = AutoModelForCausalLM.from_pretrained("./gpt2model")
model2.load_state_dict(torch.load(model_path, map_location=device))
if model2 is None:
    raise AttributeError("Failed to load trained model")
model1 = model1.to(device)
model2 = model2.to(device)


# Load tokenizer from same base model
tokenizer = AutoTokenizer.from_pretrained("./gpt2model")
if tokenizer is None:
    raise AttributeError("Failed to load tokenizer")

# Set up tokenizer padding
tokenizer.pad_token = tokenizer.eos_token

# Sample prompt for testing
prompt = "The history of artificial intelligence began in"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
input_ids = inputs["input_ids"].to(device)


# Function to generate tokens using both models
def generate_tokens(model1, model2, input_ids, n):
    generated = []
    i = 0
    eos_reached = False
    with torch.no_grad():
        while i < n and not eos_reached:
            with ThreadPoolExecutor() as executor:
                future1 = executor.submit(model1, input_ids=input_ids)
                future2 = executor.submit(model2, input_ids=input_ids)
                output1 = future1.result()
                output2 = future2.result()
            next_token = torch.argmax(output1.logits[:, -1, :]).item()
            secNext = torch.argmax(output2.logits[:, -1, :]).item()
            if next_token == tokenizer.eos_token_id:
                eos_reached = True
            elif secNext == tokenizer.eos_token_id:
                eos_reached = True
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.tensor([next_token], device=device).reshape(1, -1),
                    ),
                    dim=1,
                )
            else:
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.tensor([next_token, secNext], device=device).reshape(
                            1, -1
                        ),
                    ),
                    dim=1,
                )
            i += 2
    return input_ids


generated_tokens = generate_tokens(model1, model2, input_ids, 100)
generated_text = tokenizer.decode(generated_tokens[0])


print("\nInput prompt:", prompt)
print("\nGenerated text:", generated_text)
