from myModel import *
import torch
import torch

# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

def hook(module, input, output):
    print(output.shape)
    return None

model = CNN()
for module in model.children():
    module.register_forward_hook(hook)
    
output = model(torch.rand(1, 3, 28, 28))
# print(output.shape)
# input = torch.rand(3,3,3)
# # print(input.shape)
# # print(input)
# output = resnetEncoder(input)
# print(resnetEncoder.model)
# print(output.shape)