import torch
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def to_var(x):
    if type(x).__module__ == 'numpy':
        x = torch.from_numpy(x)
    elif isinstance(x, (int, list)):
        x = torch.tensor(x)
    if device != 'cpu':
        x = x.cuda()
    return x
