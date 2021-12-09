import numpy as np
from numpy.random.mtrand import shuffle
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision


class RandomDataset(Dataset):
    def __init__(self, path: str, data: np.ndarray) -> None:
        super().__init__()
        self.path = path
        self.data = data

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


def get_dataloader(path:str, split:float, batch_size : int = 32) -> DataLoader:
    '''
    Create and return DataLoader for random data.
    
    Args:
        path: str = path where all the data is contained. It assumes it follows the 
        create_random.py script structure - dataset.npy and all .npy in the same
        folder.

        split: float = percentage for spliting between train and validation.
        The number should be between 0 and 1.

        batch_size: int = mini batch size for the experiments.

    Return:
        Train and Validation DataLoaders. If split is 0, validation DataLoader will be None.
    '''
    if split > 1 or split < 0:
        raise Exception('Split should be between 0 and 1')

    dataset = np.load(f'{path}dataset.npy', allow_pickle=True)
    if split > 0:
        idx = int(dataset.shape[0] * split)
        train = dataset[:idx]
        eval = dataset[idx:]
    else:
        train = dataset
        eval = None

    train_dataloader = DataLoader(
        RandomDataset(path, train),
        batch_size=batch_size,
        shuffle=True
    )

    eval_dataloader = DataLoader(
        RandomDataset(path, eval),
        batch_size=batch_size,
        shuffle=True
    ) if split > 0 else None

    return train_dataloader, eval_dataloader
