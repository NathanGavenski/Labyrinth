import numpy as np
from tqdm import tqdm

# CartPole cart position [0] Pole angle [2]
# MountainCar

if __name__ == "__main__":
    x = np.load('./Expert/expert_CartPole-v1.npz')
    begginings = np.where(x['episode_starts'] == True)[0]
    ends = np.append(begginings[1:], x['episode_starts'].shape[0])

    dataset = np.ndarray(shape=[0, 500, x['obs'].shape[-1]])
    for beggining, end in zip(begginings, tqdm(ends)):
        idxs = [idx for idx in range(beggining, end)]
        dataset = np.append(dataset, x['obs'][idxs][None], axis=0)

    np.save('cartpole_epsidoes', dataset)
