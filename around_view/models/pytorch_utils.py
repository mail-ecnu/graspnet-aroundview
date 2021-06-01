import torch
from torch.autograd import Variable


def to_var(x):
    if type(x).__module__ == 'numpy':
        x = torch.from_numpy(x)
    elif isinstance(x, (int, list)):
        x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x
