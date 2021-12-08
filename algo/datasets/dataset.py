from collections import defaultdict
from copy import copy

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ExpertDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path
        
        with open(f'{self.path}dataset.txt', 'r') as f:
            for line in f:
                print(line)
                exit()

if __name__ == '__main__':
    ExpertDataset('./dataset/dataset10/')
