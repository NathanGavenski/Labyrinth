from tqdm import tqdm

from algo.il.datasets import get_expert_loader

if __name__ == '__main__':
    train = get_expert_loader('./dataset/dataset5/', 32)

    for mini_batch in tqdm(train):
        continue
