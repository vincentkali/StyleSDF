import torch

def my_collate(batch):
    data, target = list(), list()
    for b in batch:
        data.append(b[0])
        target.append(b[1])
    data = torch.stack(data,dim=0)
    target = torch.stack(target,dim=0)
    return data, target