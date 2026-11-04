"""Labyrinth-v0_5x5 dataset, labyrinth structures and the shared image bank.

The HuggingFace dataset ships a loading script that resolves image paths with
``str.split("/")``, which breaks on Windows. Instead, the two archives behind that
script are downloaded at a pinned revision and read the same way the script does
(one ``<split>_shortest_route.jsonl`` per split plus ``images/<split>/<n>.jpg``).
Transitions are then built with IL-Datasets' ``fn_create_dataset``.

Every observation is stored once in an ``ImageBank`` (uint8, CHW, 64x64) and
datasets/buffers hold bank indices. Models scale pixels to [0, 1] themselves.
"""
import ast
import hashlib
import json
import os
import tarfile
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from imitation_datasets.dataset.dataset import fn_create_dataset
from labyrinth.file_utils import convert_from_file

from . import ROOT

HF_REPO = "NathanGavenski/Labyrinth-v0_5x5"
HF_REVISION = "99f7955a005abe38f3860a3c826192ae4e489a24"
# Our split name -> (jsonl prefix, image folder) inside the archives.
SPLITS = {"train": "train", "eval": "eval", "test": "test", "random": "random"}
IMAGE_SIZE = 64

_resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))


def preprocess(image: Any) -> np.ndarray:
    """Resize an RGB image (PIL or HxWx3 uint8 array) to a uint8 3x64x64 array."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = _resize(image.convert("RGB"))
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------
def canonical_key(edges: List[Any], start: Any, end: Any) -> str:
    """Hash a configuration: undirected wall-free edges, start tile and goal tile."""
    canonical = sorted({tuple(sorted(int(v) for v in edge)) for edge in edges})
    payload = repr((canonical, tuple(int(v) for v in start), tuple(int(v) for v in end)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def env_key(env: Any) -> str:
    """Hash the configuration currently loaded in a Labyrinth environment."""
    env = env.unwrapped
    return canonical_key(env._pathways, env.start, env.end)  # pylint: disable=protected-access


def wall_vector(info: str) -> np.ndarray:
    """Binary vector of all interior wall slots (1 = wall) of a labyrinth-language file."""
    lines = info.split("labyrinth:")[1].strip().splitlines()
    grid = [line[2:-1:2] for line in lines[1:-2]]
    walls = []
    for row_idx, row in enumerate(grid):
        for col_idx, token in enumerate(row):
            if (row_idx % 2 == 0 and col_idx % 2 == 1) or (row_idx % 2 == 1 and col_idx % 2 == 0):
                walls.append(token in "|-")
    return np.array(walls, dtype=bool)


@dataclass(frozen=True)
class Structure:
    """One labyrinth configuration (walls, start and goal)."""
    info: str
    structure: str
    variables: Dict[str, bool] = field(hash=False, compare=False)
    key: str = ""

    @classmethod
    def from_info(cls, info: str) -> "Structure":
        """Parse labyrinth-language text with labyrinth.file_utils.convert_from_file."""
        with tempfile.NamedTemporaryFile(
            "w", suffix=".labyrinth", delete=False, encoding="utf-8"
        ) as _file:
            _file.write(info)
            path = _file.name
        try:
            structure, variables = convert_from_file(path)
        finally:
            os.remove(path)
        if any(variables.get(name, False) for name in ("key_and_lock", "icy_floor", "occlusion")):
            raise ValueError("Ablation uses the plain navigation setting only.")
        edges, start, end = structure.split(";")[:3]
        key = canonical_key(ast.literal_eval(edges), ast.literal_eval(start), ast.literal_eval(end))
        return cls(info=info, structure=structure, variables=dict(variables), key=key)

    @classmethod
    def from_file(cls, path: str) -> "Structure":
        """Read a .labyrinth file."""
        with open(path, encoding="utf-8") as _file:
            return cls.from_info(_file.read())

    @property
    def walls(self) -> np.ndarray:
        """Interior wall slots."""
        return wall_vector(self.info)


def wall_similarity(first: Structure, second: Structure) -> float:
    """Fraction of identical interior wall slots between two structures."""
    return float((first.walls == second.walls).mean())


# ---------------------------------------------------------------------------
# Image bank and indexed datasets
# ---------------------------------------------------------------------------
class ImageBank:
    """Deduplicated store of preprocessed observations, addressed by integer index."""

    def __init__(self) -> None:
        self.images: List[np.ndarray] = []
        self.index: Dict[Hashable, int] = {}
        self._stacked: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.images)

    def add(self, key: Hashable, make_image) -> int:
        """Return the index for ``key``, calling ``make_image()`` only on a miss."""
        idx = self.index.get(key)
        if idx is None:
            idx = len(self.images)
            self.images.append(make_image())
            self.index[key] = idx
            self._stacked = None
        return idx

    def tensor(self) -> torch.Tensor:
        """All images as one uint8 tensor (N, 3, 64, 64)."""
        if self._stacked is None or self._stacked.shape[0] != len(self.images):
            self._stacked = torch.from_numpy(np.stack(self.images))
        return self._stacked

    def __getitem__(self, idx: Any) -> torch.Tensor:
        return self.tensor()[torch.as_tensor(idx, dtype=torch.long)]

    def state_dict(self) -> Dict[str, Any]:
        return {"images": self.tensor().clone(), "keys": list(self.index.items())}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.images = [image.numpy() for image in state["images"]]
        self.index = dict(state["keys"])
        self._stacked = None


class IndexedTransitions(Dataset):
    """Transitions whose states are ImageBank indices.

    Exposes ``states``, ``actions``, ``next_states`` and ``dones`` tensors like
    IL-Datasets' BaselineDataset, so BCO/IUPE's ``_append_samples`` can extend them
    with ``torch.cat``. Items are (state uint8, action, next_state uint8, done).
    """

    def __init__(
        self,
        bank: ImageBank,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.bank = bank
        self.states = states.long().view(-1, 1)
        self.actions = actions.long().view(-1, 1)
        self.next_states = next_states.long().view(-1, 1)
        self.dones = dones.bool().view(-1)

    def __len__(self) -> int:
        return self.states.shape[0]

    def __getitem__(self, index: int):
        images = self.bank.tensor()
        return (
            images[self.states[index, 0]],
            self.actions[index],
            images[self.next_states[index, 0]],
            self.dones[index],
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states, "actions": self.actions,
            "next_states": self.next_states, "dones": self.dones,
        }

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.states = state["states"]
        self.actions = state["actions"]
        self.next_states = state["next_states"]
        self.dones = state["dones"]


# ---------------------------------------------------------------------------
# HuggingFace split loading
# ---------------------------------------------------------------------------
def download(data_dir: Optional[str] = None) -> str:
    """Download and extract the pinned dataset archives; returns the extraction folder."""
    from huggingface_hub import hf_hub_download  # pylint: disable=import-outside-toplevel

    data_dir = data_dir or os.path.join(ROOT, "dataset", "Labyrinth-v0_5x5", HF_REVISION)
    marker = os.path.join(data_dir, ".extracted")
    if os.path.exists(marker):
        return data_dir
    os.makedirs(data_dir, exist_ok=True)
    for name in ("dataset.tar.gz", "images.tar.gz"):
        archive = hf_hub_download(HF_REPO, name, repo_type="dataset", revision=HF_REVISION)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(data_dir, filter="data")
    with open(marker, "w", encoding="utf-8") as _file:
        _file.write(HF_REVISION)
    return data_dir


@dataclass
class SplitData:
    """Rows of one split plus the transitions IL-Datasets builds from them."""
    name: str
    rows: List[Dict[str, Any]]
    image_paths: List[str]
    structures: List[Structure]                 # unique, in order of first appearance
    row_structure: np.ndarray                   # row -> index into structures
    transition_rows: torch.Tensor               # (M,) row of each transition's state
    transition_next_rows: torch.Tensor          # (M,) row of each transition's next state
    actions: torch.Tensor                       # (M, 1)
    dones: torch.Tensor                         # (M,)

    @property
    def transition_structure(self) -> np.ndarray:
        return self.row_structure[self.transition_rows.numpy()]


def load_split(name: str, data_dir: str) -> SplitData:
    """Read one split and build (s, a, s', done) transitions with fn_create_dataset."""
    prefix = SPLITS[name]
    with open(os.path.join(data_dir, f"{prefix}_shortest_route.jsonl"), encoding="utf-8") as _file:
        rows = [json.loads(line) for line in _file]
    image_dir = os.path.join(data_dir, "images", prefix)
    image_paths = [os.path.join(image_dir, row["obs"].split(".")[0] + ".jpg") for row in rows]
    missing = [path for path in image_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing in split {name}, e.g. {missing[0]}")

    structures, by_info, row_structure = [], {}, []
    for row in rows:
        if row["info"] not in by_info:
            by_info[row["info"]] = len(structures)
            structures.append(Structure.from_info(row["info"]))
        row_structure.append(by_info[row["info"]])

    # Feed row numbers as observations so every transition maps back to its row.
    data = {
        "obs": np.arange(len(rows), dtype=np.int64).reshape(-1, 1),
        "actions": np.array([row["actions"] for row in rows], dtype=np.int64),
        "episode_starts": np.array([row["episode_starts"] for row in rows], dtype=np.int64),
    }
    states, actions, next_states, dones = fn_create_dataset(data, None, name)
    return SplitData(
        name=name,
        rows=rows,
        image_paths=image_paths,
        structures=structures,
        row_structure=np.array(row_structure),
        transition_rows=states.view(-1).long(),
        transition_next_rows=next_states.view(-1).long(),
        actions=actions.long().view(-1, 1),
        dones=dones.bool().view(-1),
    )


def split_to_transitions(split: SplitData, bank: ImageBank) -> IndexedTransitions:
    """Load a split's images into the bank and return its transitions as bank indices."""
    row_to_bank = {}

    def bank_index(row: int) -> int:
        if row not in row_to_bank:
            path = split.image_paths[row]
            row_to_bank[row] = bank.add(("file", path), lambda: preprocess(Image.open(path)))
        return row_to_bank[row]

    states = torch.tensor([bank_index(int(row)) for row in split.transition_rows])
    next_states = torch.tensor([bank_index(int(row)) for row in split.transition_next_rows])
    return IndexedTransitions(bank, states, split.actions, next_states, split.dones)
