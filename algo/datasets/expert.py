import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class ExpertDataset(Dataset):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.data = np.load(f'{path}dataset.npy', allow_pickle=True)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        _, _, state, action, next_state = self.data[index, :]
        state_image = torchvision.transforms.ToTensor()(
            np.load(f'{self.path}{int(state)}.npy', allow_pickle=True)
        )
        next_state_image = torchvision.transforms.ToTensor()(
            np.load(f'{self.path}{int(next_state)}.npy', allow_pickle=True)
        )
        action = torch.tensor(action.astype(int))
        return state_image, next_state_image, action


def get_dataloader(path, batch_size=32):
    return DataLoader(
        ExpertDataset(path),
        batch_size=batch_size,
        shuffle=True
    )
