import torch
from PIL import Image
from utils.tens import transpose
import numpy as np


def show_tensor(input: torch.Tensor, filename: str):
    input_numpy: np.ndarray = input.mul(255).to(torch.uint8).cpu().numpy()
    input_trans: torch.Tensor = transpose(input_numpy, "CHW", "HWC")
    image: Image.Image = Image.fromarray(input_trans, mode="RGB")
    image.save(filename)
    exit(0)
