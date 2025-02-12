# fit reward model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn

data_path = "./data"
model_path = "./gpt2model"
device = "cuda"


steps = 20


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
            if output is None or output.logits is None or output.hidden_states is None:
                raise AttributeError("Model output or its attributes are None")
            logits = output.logits[0, -1, :].squeeze()
            logits = torch.softmax(logits, dim=0).tolist()
            next_token = torch.argmax(output.logits[0, -1, :]).item()
            full_logits.append(logits)
            input_ids = torch.cat(
                (input_ids, torch.tensor([[next_token]]).to(device)), dim=-1
            )
            input_ids = input_ids.to(device)
            final_token_hidden_states.append(output.hidden_states[-1][0, -1, :])
            del output
    return tokens["input_ids"], input_ids, full_logits, final_token_hidden_states


class RewardModel(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        token_embedding_size=768,
        vocab_size=50257,
        dropout_prob=0.2,
    ):  # outputs a value between 0 and 1, prob estimation
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(
            hidden_size + token_embedding_size, hidden_size + token_embedding_size
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size + token_embedding_size, 1)

    def forward(self, hidden_state, token_embedding):
        x = torch.cat([hidden_state, token_embedding], dim=-1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


def trainRewardModel(model, batched_trajectory_data, llm_model, optimizer):
    model.train()
    loss_function = nn.MSELoss()
    loss = 0
    embedding_layer = llm_model.transformer.wte
    embedding_layer = embedding_layer.to(device)
    optimizer.zero_grad()
    for batch in batched_trajectory_data:
        prompt_ids, full_ids, full_logits, final_hidden_states = batch
        logits = torch.tensor(full_logits).to(device)  # (steps,vocab_size)
        hidden_states = torch.stack(final_hidden_states).to(
            device
        )  # (steps,hidden_size)
        new_toks = full_ids.shape[1] - prompt_ids.shape[1]
        for i in range(new_toks - 1):  # i is the token context
            target_token = full_ids[
                0, prompt_ids.shape[1] + i + 1
            ]  # two tokens in advance
            hidden_state = hidden_states[i]
            targ_probs = logits[i + 1, :]  # label
            # Get 5 random token indices and combine with target token
            # Get 5 random indices from full vocab
            random_indices = torch.randint(0, targ_probs.shape[-1], (5,)).to(device)
            # Get max probability token index
            max_index = torch.argmax(targ_probs).unsqueeze(0).to(device)
            # Get indices of top 50 probabilities and sample 3 randomly from them
            top_50_probs_indices = torch.topk(targ_probs, 50, dim=0)[1]
            top_50_random = top_50_probs_indices[torch.randint(0, 50, (3,))].to(device)
            # Combine all indices
            random_indices = torch.cat([random_indices, max_index, top_50_random])
            all_tokens = torch.cat(
                [random_indices, target_token.unsqueeze(0)]
            ).squeeze()
            all_tokens = all_tokens.to(device)

            # Get embeddings and predictions for all tokens at once
            all_embeddings = embedding_layer(all_tokens)
            all_preds = torch.cat([model(hidden_state, emb) for emb in all_embeddings])
            loss += loss_function(all_preds, targ_probs[all_tokens])
            """target_token = target_token.to(device)
            token_embedding = embedding_layer(target_token)
            token_embedding = token_embedding.to(device)
            hidden_state = hidden_state.to(device)
            predicted_tok_prob = model(hidden_state, token_embedding)
            loss += loss_function(predicted_tok_prob, targ_probs[target_token])"""
    print(f"Batch Loss: {loss}")
    with open("reward_loss.txt", "a") as f:
        f.write(f"Batch Loss: {loss}\n")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


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


def reward_training_loop(reward_model, llm_model, traj_data_list):
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    reward_model = reward_model.to(device)
    batched_traj_data_list = [
        traj_data_list[i : i + 15] for i in range(0, len(traj_data_list), 15)
    ]
    for batched_traj_data in batched_traj_data_list:
        trainRewardModel(reward_model, batched_traj_data, llm_model, optimizer)


def train_llm(
    reward_model, base_gpt, train_gpt, tokenizer, prompts, beta=0.5, epochs=30, lr=1e-4
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
                f"Epoch {epoch+1}/{epochs} Batch {count}/{len(prompts)}: {-loss.item()}"
            )
            """with open("train_gpt_loss.txt", "a") as file:
                file.write(
                    f"Epoch {epoch+1}/{epochs} Batch {count}/{len(prompts)}: {loss.item()}"
                )"""
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}/{epochs} completed")
        """with open("train_gpt_loss.txt", "a") as file:
            file.write(f"Epoch {epoch+1}/{epochs} completed")"""
    return train_gpt
