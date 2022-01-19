from os import path
from algo.il import ILPO, create_ilpo_dataset

if __name__ == '__main__':
    if not path.exists('./dataset/ilpo_dataset'):
        create_ilpo_dataset(
            path='./dataset/dataset5',
            file='dataset.npy',
            output_dir='./dataset/ilpo_dataset',
        )

    ilpo = ILPO(
        input_dir='./dataset/ilpo_dataset',
        output_dir='./tmp/ilpo/output',
        checkpoint_dir='./tmp/ilpo/checkpoint',
        batch_size=4,
        ngf=15,
    )
    ilpo.run()
