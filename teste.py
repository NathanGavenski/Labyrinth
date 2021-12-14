from tqdm import tqdm

from algo.il.datasets import get_random_loader

if __name__ == '__main__':
    train, valid = get_random_loader('./dataset/random_dataset5/', .7, 32)

    for mini_batch in tqdm(train):
        continue

    for mini_batch in tqdm(valid):
        continue