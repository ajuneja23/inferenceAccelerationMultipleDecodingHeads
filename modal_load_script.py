import modal
from modelTrainLoop import model_train_loop
from pathlib import Path, PosixPath
import torch

app = modal.App("train-run-v1")
volume = modal.Volume.from_name("model-store", create_if_missing=True)
image = modal.Image.debian_slim()
image = image.pip_install("transformers[torch]")
image = image.pip_install("datasets")


@app.function(image=image, timeout=10800, volumes={"/model_dir": volume})
def load_model():
    load_path = "/model_dir/final_trained_model.pth"
    model = torch.load(load_path, map_location=torch.device("cpu"), weights_only=True)
    save_path = "train_gpt.pth"
    torch.save(model, save_path)


@app.local_entrypoint()
def main():
    load_model.remote()
