import torch

weight = torch.load('./output/model_final.pth')

print(weight.keys())