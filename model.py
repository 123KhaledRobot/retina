import torch
import torch.nn as nn
import torchvision.models as models

weights=models.ResNet50_Weights.DEFAULT
model =models.resnet50(weights=weights)

model.eval()

for param in model.parameters():
  param.requires_grad=False

num_classes=5

num_features=model.fc.in_features
model.fc=nn.Linear(num_features,num_classes)

print(model)

for name,param in model.named_parameters():
  print(name,param.requires_grad)