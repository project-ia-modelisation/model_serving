import torch

model = torch.nn.Linear(10, 1)
model.load_state_dict(torch.load("../model-training/data/model.pth"))