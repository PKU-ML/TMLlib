from utils.tens import normalize
import torch


class NormalModel(torch.nn.Module):

    def __init__(self, model: torch.nn.Module, normal_model: bool = True) -> None:
        super().__init__()
        self.model = model
        self.normal_model = normal_model

    def forward(self, x):
        if self.normal_model:
            return self.model(normalize(x))
        else:
            return self.model(x)
