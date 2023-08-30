from collections import defaultdict
import numpy as np
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
sns.set_theme()

# CartPole cart position [0] Pole angle [2]
# MountainCar position [0] and velocity [1]

if __name__ == "__main__":
    x = np.load('./tmp/experts/mountaincar/teacher.npz', allow_pickle=True)
    print(list(x.keys()))
    for k in x.keys():
        print(k, x[k].shape)

    X_embedded = x['states']
    # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', n_jobs=4, verbose=True).fit_transform(x['states'])
    # np.save('./tmp/experts/embedded_Antv3', X_embedded)
    # X_embedded = np.load('./tmp/experts/walker/embedded_teacher.npy', allow_pickle=True)

    rewards = defaultdict(int)
    for reward in x['episode_returns']:
        rewards[reward] += 1
        
    begginings = np.where(x['episode_starts'] == True)[0]
    ends = np.append(begginings[1:], x['episode_starts'].shape[0])
    split = int(begginings.shape[0] // 2)
    print("Amount for each:", split)

    train_b, train_e = begginings[:split], ends[:split]
    eval_b, eval_e = begginings[split:], ends[split:]

    episodes = []
    for b, e in zip([train_b, eval_b], [train_e, eval_e]):
        dataset = np.ndarray(shape=[0, X_embedded.shape[-1]])
        for beggining, end in zip(b, e):
            idxs = [idx for idx in range(beggining, end)]
            dataset = np.append(dataset, X_embedded[idxs], axis=0)
        episodes.append(dataset)

    # ########### Plot ########### #

    x_idx = 0
    y_idx = 1
    datas = []
    for episode in episodes:
        size = 50
        episode[:, x_idx] = (episode[:, x_idx] - episode[:, x_idx].min()) / (episode[:, x_idx].max() - episode[:, x_idx].min())
        episode[:, y_idx] = (episode[:, y_idx] - episode[:, y_idx].min()) / (episode[:, y_idx].max() - episode[:, y_idx].min())

        x_max, x_min = episode[:,x_idx].max(), episode[:,x_idx].min()
        y_max, y_min = episode[:,y_idx].max(), episode[:,y_idx].min()

        x_ratio = abs((x_max - x_min) / size)
        y_ratio = abs((y_max - y_min) / size)

        data = np.zeros(shape=(size+1, size+1))
        for state in episode:
            x = int(state[x_idx] / x_ratio)
            y = int(state[y_idx] / y_ratio)
            data[x, y] += 1

        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        datas.append(data)

    fig = plt.figure(figsize=(20, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax = sns.heatmap(
        datas[0],
        annot=False,
        fmt=".0%",
        annot_kws={"size": 10},
        linewidths=.5,
        ax=ax1
    )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Train set')

    ax = sns.heatmap(
        datas[1],
        annot=False,
        fmt=".0%",
        annot_kws={"size": 10},
        linewidths=.5,
        ax=ax2
    )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Evaluation set')
    
    ax = sns.heatmap(
        np.abs(datas[0] - datas[1]),
        annot=False,
        fmt=".0%",
        annot_kws={"size": 10},
        linewidths=.5,
        ax=ax3
    )
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('Difference')
    plt.show()