"""Encoding analysis: which training state does each evaluation state look like?

For every state the agent visits on the eval/test labyrinths, find the nearest
training-dataset state in the policy's penultimate-layer feature space (cosine and
L2) and record whether
    (a) both states have the agent on the same tile,
    (b) the neighbour's (expert) action is optimal in the visited state,
    (c) the policy's chosen action equals the neighbour's action,
plus the wall similarity between the two labyrinths (fraction of identical interior
wall slots in their ``info`` configurations).
"""
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .data import IndexedTransitions, SplitData, Structure
from .envs import LabyrinthImageEnv

METRICS = ("cosine", "l2")
FLAGS = ("same_tile", "nn_action_optimal", "policy_matches_nn", "wall_similarity")


def train_state_info(split: SplitData, transitions: IndexedTransitions, env: LabyrinthImageEnv) -> Dict[str, np.ndarray]:
    """Bank index, action, structure and agent tile for every training transition.

    Tiles are recovered by replaying each dataset episode in the environment; the
    replay must end on the goal, which also checks dataset/environment agreement.
    """
    rows = split.transition_rows.numpy()
    tiles = np.zeros(len(rows), dtype=np.int64)
    structure_of_row = split.row_structure
    position = 0
    while position < len(rows):
        structure = split.structures[structure_of_row[rows[position]]]
        env.load_structure(structure)
        terminated = False
        while True:
            tiles[position] = env.agent_tile
            _, _, terminated, truncated, _ = env.step(int(transitions.actions[position, 0]))
            done = bool(transitions.dones[position])
            position += 1
            if done or terminated or truncated:
                break
        if not (terminated and done):
            raise RuntimeError(f"Dataset episode ending at transition {position} does not reach the goal.")
    return {
        "bank": transitions.states.view(-1).numpy().copy(),
        "action": transitions.actions.view(-1).numpy().copy(),
        "structure": structure_of_row[rows],
        "tile": tiles,
    }


def _features(method: Any, bank, indices: np.ndarray, batch: int = 1024) -> torch.Tensor:
    images = bank.tensor()
    out = []
    for start in range(0, len(indices), batch):
        idx = torch.from_numpy(indices[start:start + batch])
        out.append(method.features(images[idx]).float().cpu())
    return torch.cat(out)


def encoding_analysis(
    method: Any,
    bank,
    train: Dict[str, np.ndarray],
    train_structures: Sequence[Structure],
    visited: List[Dict[str, Any]],
    split_structures: Dict[str, Sequence[Structure]],
):
    """Return (per-state DataFrame, per-split summary dict)."""
    if not visited:
        return pd.DataFrame(), {}
    train_features = _features(method, bank, train["bank"])
    # Identical images have identical features, so neighbours are found per unique image.
    unique_visited, inverse = np.unique(
        np.array([v["bank"] for v in visited], dtype=np.int64), return_inverse=True
    )
    visited_features = _features(method, bank, unique_visited)

    neighbours = {}
    with torch.no_grad():
        cos = F.normalize(visited_features, dim=1) @ F.normalize(train_features, dim=1).T
        neighbours["cosine"] = (cos.argmax(1).numpy()[inverse], 1 - cos.max(1).values.numpy()[inverse])
        dist = torch.cdist(visited_features, train_features)
        neighbours["l2"] = (dist.argmin(1).numpy()[inverse], dist.min(1).values.numpy()[inverse])

    train_walls = np.stack([s.walls for s in train_structures])
    walls = {
        split: np.stack([s.walls for s in structures])
        for split, structures in split_structures.items()
    }

    frame = pd.DataFrame({
        "split": [v["split"] for v in visited],
        "labyrinth": [v["labyrinth"] for v in visited],
        "step": [v["step"] for v in visited],
        "tile": [v["tile"] for v in visited],
        "policy_action": [v["action"] for v in visited],
        "expert_action": [v["expert_action"] for v in visited],
        "optimal_actions": [",".join(map(str, v["optimal_actions"])) for v in visited],
    })
    frame["policy_action_optimal"] = [v["action"] in v["optimal_actions"] for v in visited]
    for metric in METRICS:
        nn_idx, distance = neighbours[metric]
        nn_action = train["action"][nn_idx]
        nn_structure = train["structure"][nn_idx]
        frame[f"{metric}_nn_index"] = nn_idx
        frame[f"{metric}_nn_structure"] = nn_structure
        frame[f"{metric}_nn_tile"] = train["tile"][nn_idx]
        frame[f"{metric}_nn_action"] = nn_action
        frame[f"{metric}_distance"] = distance
        frame[f"{metric}_same_tile"] = frame["tile"].to_numpy() == train["tile"][nn_idx]
        frame[f"{metric}_nn_action_optimal"] = [
            int(a) in v["optimal_actions"] for a, v in zip(nn_action, visited)
        ]
        frame[f"{metric}_policy_matches_nn"] = frame["policy_action"].to_numpy() == nn_action
        frame[f"{metric}_wall_similarity"] = [
            float((walls[v["split"]][v["labyrinth"]] == train_walls[s]).mean())
            for v, s in zip(visited, nn_structure)
        ]

    summary = {}
    for split, group in frame.groupby("split"):
        summary[split] = {
            "n_states": int(len(group)),
            "policy_action_optimal": float(group["policy_action_optimal"].mean()),
        }
        for metric in METRICS:
            summary[split][metric] = {
                flag: float(group[f"{metric}_{flag}"].mean()) for flag in FLAGS
            }
    return frame, summary


def save_frame(frame: pd.DataFrame, path_without_suffix: str) -> str:
    """Write parquet, or CSV when no parquet engine is available."""
    try:
        path = f"{path_without_suffix}.parquet"
        frame.to_parquet(path, index=False)
    except (ImportError, ValueError):
        path = f"{path_without_suffix}.csv"
        frame.to_csv(path, index=False)
    return path
