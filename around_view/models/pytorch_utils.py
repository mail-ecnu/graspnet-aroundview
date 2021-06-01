import torch
from torch.autograd import Variable


def to_var(x):
    if type(x).__module__ == 'numpy':
        x = torch.from_numpy(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x
