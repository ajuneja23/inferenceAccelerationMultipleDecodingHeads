import modal
from modelTrainLoop import model_train_loop
from pathlib import Path, PosixPath
import torch

app = modal.App("train-run-v1")
volume = modal.Volume.from_name("model-store", create_if_missing=True)
image = modal.Image.debian_slim()
image = image.pip_install("transformers[torch]")
image = image.pip_install("datasets")
image = image.add_local_dir("./gpt2model", remote_path="/gpt2model")
image = image.add_local_file(
    "./trained_reward_model.pth", remote_path="/trained_reward_model.pth"
)


@app.function(gpu="A10G", image=image, timeout=10800, volumes={"/model_dir": volume})
def run_model_training():
    model = model_train_loop()
    save_path = "/model_dir/final_trained_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    volume.commit()


@app.local_entrypoint()
def main():
    run_model_training.remote()
