import torch as T
import torch.nn as nn
import torch.nn.functional as F

class ICM(nn.Module):
    def __init__(self, input_dims, n_actions=2, )