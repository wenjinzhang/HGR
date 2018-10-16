import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(x)
x = x.resize_(1, 2, 3)
print(x)

