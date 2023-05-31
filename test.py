import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

a = torch.tensor(
    [
        0.0881,
        0.0817,
        0.0999,
        0.0691,
        0.0773,
        0.0684,
        0.0793,
        0.1011,
        0.0741,
        0.0892,
        0.0818,
        0.0899,
    ]
)
print(sum(a))
