# Accelerating Inference via Multiple Decoding Heads

## Introduction

This project explores techniques to accelerate inference in machine learning models by utilizing multiple decoding heads. We focus on the Medusa technique and our novel token-based reinforcement learning (RL) methods to enhance the efficiency and speed of model inference.

## Medusa Technique

The Medusa technique involves using multiple decoding heads in a language model to generate multiple tokens at a time. This parallel processing approach reduces the overall inference time. Key benefits include:

- **Increased Throughput**: By processing multiple tokens at once, the model can handle larger inputs more efficiently.
- **Reduced Latency**: Parallel decoding reduces the time required to generate outputs, making the model faster.

## Token-Based RL Methods

Our token-based reinforcement learning methods further optimize inference by combining current research in utilizing RL to teach models at a token-level in this inference acceleration environment. Specifically, we fitted a reward model to predict the probability of a token given the hidden state and token embedding. We then used Proximal Policy Optimization (PPO) with a reward minus beta times KL divergence formulation as bandits to train the GPT model to predict the next tokens.
In our approach, the reward model is designed to predict the probability of a token by utilizing the last hidden state of the first `n` tokens plus a proposed token embedding. The reward model then outputs the probability that the token corresponding to this token embedding is the `(n+2)`-th token in the sequence. This is done to fit the second head so during inference time both heads can be executed concurrently to generate tokens faster, as we generate two at a time rather than one at a time.
