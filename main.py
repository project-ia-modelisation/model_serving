import torch

from model_training.data import model

models = torch.nn.Linear(10, 1)
models.load_state_dict(torch.load(model))