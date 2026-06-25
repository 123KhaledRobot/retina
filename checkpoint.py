import torch
import torch.nn as nn
from torchvision import models

model = models.resnet50(weights=None)
model.fc = nn.Linear(2048, 5)  # rebuild architecture
model.load_state_dict(torch.load("retina_weights.pth"))
