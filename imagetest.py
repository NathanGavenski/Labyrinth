import datasets
import json
from os import listdir
from os.path import isfile, join


_CITATION = ""

_DESCRIPTION = "Dataset for training agents with Maze-v0 environment."

_HOMEPAGE = "https://huggingface.co/datasets/NathanGavenski/imagetest"

_LICENSE = ""

_REPO = "https://huggingface.co/datasets/NathanGavenski/imagetest"


class ImageSet(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "obs": datasets.Value("string"),
                "actions": datasets.Value("int32"),
                "rewards": datasets.Value("float32"),
                "episode_starts": datasets.Value("bool"),
                "maze": datasets.Value("string"),
            }),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        image_path = dl_manager.download_and_extract(f"{_REPO}/resolve/main/images.tar.gz")
        info_path = dl_manager.download_and_extract(f"{_REPO}/resolve/main/dataset.tar.gz")
        return [
            datasets.SplitGenerator(
                name="all_routes",
                gen_kwargs={
                    "images": image_path,
                    "infos": f"{info_path}/all_routes.jsonl"
                }
            ),
            datasets.SplitGenerator(
                name="single_route",
                gen_kwargs={
                    "images": image_path,
                    "infos": f"{info_path}/single_route.jsonl"
                }
            ),
            datasets.SplitGenerator(
                name="shortest_route",
                gen_kwargs={
                    "images": image_path,
                    "infos": f"{info_path}/shortest_route.jsonl"
                }
            ),
        ]

    def _generate_examples(self, images, infos):
        images_paths = f"{images}/images"
        images = [join(images_paths, f) for f in listdir(images_paths) if isfile(join(images_paths, f))]

        images_dict = {}
        for image in images:
            images_dict[image.split("/")[-1].split(".")[0]] = image

        with open(infos, encoding="utf-8") as data:
            for idx, line in enumerate(data):
                record = json.loads(line)
                index = record["obs"].split(".")[0]
                yield idx, {
                    "obs": images_dict[index],
                    "actions": record["actions"],
                    "rewards": record["rewards"],
                    "episode_starts": record["episode_starts"],
                    "maze": record["maze"],
                }
