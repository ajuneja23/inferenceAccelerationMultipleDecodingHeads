from modelTrainLoop import model_train_loop
from pathlib import Path, PosixPath
import torch


model = model_train_loop()
save_path = "second_head.pth"
torch.save(model.state_dict(), save_path)
