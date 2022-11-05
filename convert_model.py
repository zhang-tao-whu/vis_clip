import torch

with open('./output/model_final.pth', 'r') as f:
    weight = torch.load(f)

print(weight.keys())