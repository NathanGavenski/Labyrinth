import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class ExpertDataset(Dataset):
    def __init__(self, path: str, amount: int = 1) -> None:
        super().__init__()
        self.path = path
        self.data = np.load(f'{path}dataset.npy', allow_pickle=True)
        if amount > 1:
            self.data = np.repeat(
                self.data, 
                repeats=amount,
                axis=0
            )

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index):
        _, _, state, action, next_state, _, _, _, _  = self.data[index, :]
        state_image = torchvision.transforms.ToTensor()(
            np.load(f'{self.path}{int(state)}.npy', allow_pickle=True)
        )
        next_state_image = torchvision.transforms.ToTensor()(
            np.load(f'{self.path}{int(next_state)}.npy', allow_pickle=True)
        )
        action = torch.tensor(action.astype(int))
        return state_image, next_state_image, action


def get_dataloader(path:str, batch_size: int = 32, amount: int = 10) -> DataLoader:
    '''
    Create and return DataLoader for expert data.
    
    Args:
        path: str = path where all the data is contained. It assumes it follows the 
        create_expert.py script structure - dataset.npy and all .npy in the same
        folder.

        batch_size: int = mini batch size for the experiments.

    Return:
        Train DataLoaders.
    '''
    return DataLoader(
        ExpertDataset(path, amount),
        batch_size=batch_size,
        shuffle=True
    )
