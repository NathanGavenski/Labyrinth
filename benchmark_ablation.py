"""Encoder ablation (CNN vs ResNet-18) for imitation learning on Labyrinth 5x5.

Re-runs the 5x5 Labyrinth benchmark (systematicity setting: same start and goal
tiles in every split, only the walls change) for DAgger, GAIL, BCO, SQIL and IUPE
(and optionally BC) with both policy encoders, and reports AER/SR per split plus an
encoding analysis of the learned representations.

Each (method, encoder, seed) runs in its own subprocess and folder:
    <out>/<method>/<encoder>/seed_<seed>/
        config.yaml               merged method + encoder hyperparameters
        log.txt, console.log      run log and raw stdout/stderr
        train_metrics.jsonl       per-epoch training metrics
        checkpoint.pt             periodic resumable checkpoint (removed when done)
        final.pt                  model weights after the last epoch
        results.json              per-labyrinth returns/successes, summaries, config
        encoding_analysis.parquet per-state nearest-neighbour records
and the combined <out>/summary.csv and <out>/ablation_table.tex.

Training environments can only hold train-split configurations
(ablation.envs.GuardedLabyrinth). Eval/test environments are created after training.
The model evaluated is always the one after the last epoch.

Example:
    python benchmark_ablation.py --methods dagger gail bco sqil iupe \
        --encoders cnn resnet18 --epochs 1000 --seeds 0 --out results/ablation --resume
"""
import argparse
import importlib.metadata
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import ablation  # noqa: F401  (must precede labyrinth/methods imports)
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from ablation.analysis import encoding_analysis, save_frame, train_state_info
from ablation.data import (
    HF_REPO, HF_REVISION, ImageBank, download, load_split, preprocess, split_to_transitions,
)
from ablation.envs import GuardedLabyrinth, LabyrinthImageEnv, LeakageError
from ablation.report import run_dir, write_summary, write_table
from methods import METHODS
from methods.base import TrainingContext

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METHODS = ["dagger", "gail", "bco", "sqil", "iupe"]
ENCODER_CONFIGS = {"cnn": "configs/cnn.yaml", "resnet18": "configs/resnet.yaml"}
ENCODER_KEYS = ("policy", "idm", "hidden_dim")
EXTRA_SPLITS = {"new_s0": "change_start", "new_s0_and_g": "change_start_and_goal"}
IDM_METHODS = ("bco", "iupe")

logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Arguments, logging, provenance
# ---------------------------------------------------------------------------
def get_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=sorted(METHODS))
    parser.add_argument("--encoders", nargs="+", default=["cnn", "resnet18"], choices=sorted(ENCODER_CONFIGS))
    parser.add_argument("--size", type=int, default=5, help="Labyrinth size (only 5 is available).")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="results/ablation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs with results.json and resume runs from their checkpoint.")
    parser.add_argument("--extra-splits", action="store_true",
                        help="Also evaluate on train structures with new start (and goal) positions.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ckpt-every", type=int, default=50, help="Checkpoint interval in epochs.")
    parser.add_argument("--enjoy-criteria", type=int, default=100,
                        help="BCO/IUPE: append policy rollouts at epoch 0 and every N epochs.")
    parser.add_argument("--debug-n-labyrinths", type=int, default=None,
                        help="Debug only: evaluate on the first N labyrinths per split and use N "
                             "episodes for training-time rollouts.")
    parser.add_argument("--data-dir", default=None, help="Where to extract the dataset archives.")
    parser.add_argument("--run-one", nargs=3, metavar=("METHOD", "ENCODER", "SEED"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.size != 5:
        parser.error("Only the 5x5 dataset (NathanGavenski/Labyrinth-v0_5x5) is wired up.")
    return args


def setup_logging(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )


def il_datasets_info() -> Dict[str, Any]:
    """Version and git commit of the installed IL-Datasets (must come from main)."""
    dist_name = importlib.metadata.packages_distributions()["imitation_datasets"][0]
    dist = importlib.metadata.distribution(dist_name)
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    vcs = direct.get("vcs_info", {})
    info = {
        "distribution": dist_name,
        "version": dist.version,
        "url": direct.get("url"),
        "commit": vcs.get("commit_id"),
        "requested_revision": vcs.get("requested_revision"),
    }
    if "NathanGavenski/IL-Datasets" not in str(info["url"]) or info["requested_revision"] != "main":
        raise RuntimeError(
            "IL-Datasets must be installed from "
            f"git+https://github.com/NathanGavenski/IL-Datasets.git@main, found {info}"
        )
    return info


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.tolist())
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as _file:
        json.dump(to_jsonable(payload), _file, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Data and checks
# ---------------------------------------------------------------------------
def prepare_data(args: argparse.Namespace, with_random: bool) -> SimpleNamespace:
    """Load splits, build the image bank/transitions and check split separation."""
    data_dir = download(args.data_dir)
    names = ["train", "eval", "test"] + (["random"] if with_random else [])
    splits = {name: load_split(name, data_dir) for name in names}
    keys = {name: {s.key for s in split.structures} for name, split in splits.items()}

    for name in ("train", "eval", "test"):
        if len(keys[name]) != len(splits[name].structures):
            raise RuntimeError(f"Split {name} contains duplicate structures.")
    for first, second in (("train", "eval"), ("train", "test"), ("eval", "test")):
        overlap = keys[first] & keys[second]
        if overlap:
            raise RuntimeError(f"{len(overlap)} structures shared by {first} and {second}.")
    if with_random and not keys["random"] <= keys["train"]:
        raise RuntimeError(
            f"{len(keys['random'] - keys['train'])} random-split structures are not train structures."
        )

    bank = ImageBank()
    train = split_to_transitions(splits["train"], bank)
    random_transitions = split_to_transitions(splits["random"], bank) if with_random else None
    return SimpleNamespace(splits=splits, keys=keys, bank=bank, train=train, random=random_transitions)


def episode_bounds(rows: List[Dict[str, Any]]) -> List[range]:
    starts = [i for i, row in enumerate(rows) if row["episode_starts"]] + [len(rows)]
    return [range(a, b) for a, b in zip(starts[:-1], starts[1:])]


def returns_model(env: LabyrinthImageEnv) -> Dict[str, float]:
    """Step penalty, step cap and the failure return of the unmodified environment."""
    unwrapped = env.unwrapped
    penalty = -0.1 / (unwrapped.shape[0] * unwrapped.shape[1])
    cap = unwrapped.max_episode_steps
    return {"step_penalty": penalty, "step_cap": cap, "fail_return": penalty * cap}


def optimal_return(n_steps: int, penalty: float) -> float:
    """Return of an n-step shortest path: n - 1 penalised steps, then +1 at the goal."""
    return 1.0 + penalty * (n_steps - 1)


def normalise(raw: float, optimal: float, fail: float, top: float = 1.0) -> float:
    """Affine per-labyrinth map with failure -> fail (-4.00) and shortest path -> 1.00."""
    return fail + (raw - fail) * (top - fail) / (optimal - fail)


def preflight(args: argparse.Namespace, out: str) -> Dict[str, Any]:
    """Environment/dataset checks that must pass before any training run."""
    started = time.time()
    data = prepare_data(args, with_random=True)
    bank = ImageBank()
    env = LabyrinthImageEnv(bank)
    model = returns_model(env)
    report: Dict[str, Any] = {"returns_model": model, "splits": {}}

    # 1. Guard: train configurations pass, eval/test configurations raise.
    guard = GuardedLabyrinth(bank, data.splits["train"].structures, seed=0)
    seen = []
    for _ in range(len(data.splits["train"].structures)):
        guard.reset()
        seen.append(guard.key)
    if set(seen) != data.keys["train"]:
        raise RuntimeError("Guarded resets did not cycle through exactly the train structures.")
    fired = {}
    for split in ("eval", "test"):
        try:
            guard.load_structure(data.splits[split].structures[0])
            fired[split] = False
        except LeakageError:
            fired[split] = True
    try:
        guard.reset(options={"agent": False})
        fired["random_generation"] = False
    except LeakageError:
        fired["random_generation"] = True
    if not all(fired.values()):
        raise RuntimeError(f"Leakage guard did not fire: {fired}")
    report["guard"] = {"train_cycle_ok": True, "fired": fired}

    # 2. Expert, failure and dataset agreement on every train/eval/test structure.
    for split_name in ("train", "eval", "test"):
        split = data.splits[split_name]
        expert_returns, normalised, mismatched_lengths = [], [], 0
        strict, optimal_membership, total_actions = 0, 0, 0
        for bounds in episode_bounds(split.rows):
            rows = [split.rows[i] for i in bounds]
            structure = split.structures[split.row_structure[bounds[0]]]
            env.load_structure(structure)
            path = env.shortest_path()
            n_steps = len(path) - 1
            mismatched_lengths += n_steps != len(rows) - 1
            for row in rows[:-1]:
                expert, optimal = env.optimal_actions()
                strict += expert == row["actions"]
                optimal_membership += row["actions"] in optimal
                total_actions += 1
                env.step(row["actions"])
            env.load_structure(structure)
            total, terminated = 0.0, False
            while not terminated:
                _, reward, terminated, truncated, _ = env.step(env.expert_action())
                total += reward
                if truncated:
                    break
            expected = optimal_return(n_steps, model["step_penalty"])
            if not (terminated and np.isclose(total, expected)):
                raise RuntimeError(f"Expert failed on a {split_name} structure ({total} vs {expected}).")
            expert_returns.append(total)
            normalised.append(normalise(total, expected, model["fail_return"]))

        # Forced failure: DOWN from the bottom-left start never moves.
        env.load_structure(split.structures[0])
        total, terminated, truncated = 0.0, False, False
        while not (terminated or truncated):
            _, reward, terminated, truncated, _ = env.step(2)
            total += reward
        fail_expected = optimal_return(len(env.shortest_path()) - 1, model["step_penalty"])
        failure = {
            "return": total, "terminated": bool(terminated), "truncated": bool(truncated),
            "normalised": normalise(total, fail_expected, model["fail_return"]),
        }
        if terminated or not np.isclose(total, -4.0) or not np.isclose(failure["normalised"], -4.0):
            raise RuntimeError(f"Forced failure on {split_name} does not score -4.00: {failure}")
        if mismatched_lengths or optimal_membership != total_actions:
            raise RuntimeError(
                f"{split_name}: {mismatched_lengths} episodes differ from the shortest path length, "
                f"{total_actions - optimal_membership} dataset actions are not optimal."
            )
        if not np.allclose(normalised, 1.0):
            raise RuntimeError(f"{split_name}: expert does not normalise to 1.00.")
        report["splits"][split_name] = {
            "structures": len(split.structures),
            "expert_raw_return_mean": float(np.mean(expert_returns)),
            "expert_raw_return_min": float(np.min(expert_returns)),
            "expert_raw_return_max": float(np.max(expert_returns)),
            "expert_normalised_return": float(np.mean(normalised)),
            "expert_success_rate": 100.0,
            "forced_failure": failure,
            "dataset_actions_optimal": f"{optimal_membership}/{total_actions}",
            "dataset_actions_equal_expert_tie_break": f"{strict}/{total_actions}",
        }

    # 3. Dataset images match environment renders (train/eval/test starts and random replays).
    diffs = {}
    for split_name in ("train", "eval", "test", "random"):
        split = data.splits[split_name]
        episodes = episode_bounds(split.rows)[:3 if split_name == "random" else 5]
        worst = 0.0
        for bounds in episodes:
            env.load_structure(split.structures[split.row_structure[bounds[0]]])
            rows = list(bounds) if split_name == "random" else [bounds[0]]
            for row in rows:
                image = preprocess(Image.open(split.image_paths[row])).astype(np.int16)
                worst = max(worst, float(np.abs(image - env.observe().astype(np.int16)).mean()))
                if split_name == "random" and row != rows[-1]:
                    env.step(split.rows[row]["actions"])
        diffs[split_name] = worst
    if max(diffs.values()) > 5.0:
        raise RuntimeError(f"Dataset images do not match environment renders: {diffs}")
    report["max_mean_abs_pixel_diff_dataset_vs_env"] = diffs
    report["random_split"] = {
        "structures": len(data.splits["random"].structures),
        "subset_of_train": True,
        "transitions": len(data.random),
    }
    report["split_overlap"] = {"train_eval": 0, "train_test": 0, "eval_test": 0}
    report["seconds"] = time.time() - started
    report["passed"] = True
    write_json(os.path.join(out, "preflight.json"), report)
    return report


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def write_config(method: str, encoder: str, path: str) -> Dict[str, Any]:
    """Merge the encoder config (configs/*.yaml) with the method config, if any."""
    with open(os.path.join(ROOT, ENCODER_CONFIGS[encoder]), encoding="utf-8") as _file:
        merged = dict(yaml.safe_load(_file)["Default"])
    method_config = os.path.join(ROOT, "methods", "config", f"{method}.yaml")
    if os.path.exists(method_config):
        with open(method_config, encoding="utf-8") as _file:
            own = dict(yaml.safe_load(_file)["Default"])
        encoder_values = {key: merged[key] for key in ENCODER_KEYS if key in merged}
        merged = {**own, **encoder_values}
    with open(path, "w", encoding="utf-8") as _file:
        yaml.safe_dump({"Default": merged}, _file, sort_keys=False)
    return merged


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def rng_state(generator: torch.Generator) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": generator.get_state(),
    }


def load_rng_state(state: Dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["loader"])


def evaluate(
    method: Any,
    env: LabyrinthImageEnv,
    structures: List[Any],
    split: str,
    seed: int,
    model: Dict[str, float],
    mode: Optional[str] = None,
    record: bool = False,
):
    """One greedy episode per structure. Returns (split results, visited-state records)."""
    returns, normalised, successes, steps, optimal_steps, keys = [], [], [], [], [], []
    visited: List[Dict[str, Any]] = []
    for idx, structure in enumerate(structures):
        obs, _ = env.load_structure(structure)
        if mode is not None:
            position_seed = seed * 100_003 + idx
            random.seed(position_seed)
            np.random.seed(position_seed)
            getattr(env.unwrapped, mode)()
            env.refresh_key()
            obs = env.observe()
        n_optimal = len(env.shortest_path()) - 1
        total, n_steps, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            action = method.greedy_action(obs)
            if record:
                expert, optimal = env.optimal_actions()
                visited.append({
                    "split": split, "labyrinth": idx, "step": n_steps, "bank": env.obs_index,
                    "tile": env.agent_tile, "action": action, "expert_action": expert,
                    "optimal_actions": optimal,
                })
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            n_steps += 1
        best = optimal_return(n_optimal, model["step_penalty"])
        returns.append(total)
        normalised.append(normalise(total, best, model["fail_return"]))
        successes.append(bool(terminated))
        steps.append(n_steps)
        optimal_steps.append(n_optimal)
        keys.append(env.key)

    results = {
        "n_labyrinths": len(structures),
        "aer": float(np.mean(normalised)),
        "aer_std": float(np.std(normalised)),
        "aer_raw": float(np.mean(returns)),
        "aer_raw_std": float(np.std(returns)),
        "sr": 100.0 * float(np.mean(successes)),
        "returns": returns,
        "returns_normalized": normalised,
        "successes": successes,
        "steps": steps,
        "optimal_steps": optimal_steps,
        "structure_keys": keys,
    }
    if mode is not None:
        results["position_mode"] = mode
        results["position_seed"] = f"seed * 100003 + labyrinth index (seed={seed})"
    return results, visited


def run_single(args: argparse.Namespace, method_name: str, encoder: str, seed: int) -> None:
    folder = run_dir(args.out, method_name, encoder, seed)
    os.makedirs(folder, exist_ok=True)
    setup_logging(os.path.join(folder, "log.txt"))
    started = time.time()
    logger.info("Run %s / %s / seed %d on %s", method_name, encoder, seed, args.device)
    il_info = il_datasets_info()
    seed_everything(seed)
    torch.backends.cudnn.benchmark = False

    n_debug = args.debug_n_labyrinths
    data = prepare_data(args, with_random=method_name in IDM_METHODS)
    bank = data.bank
    train_structures = data.splits["train"].structures

    config = write_config(method_name, encoder, os.path.join(folder, "config.yaml"))
    env = GuardedLabyrinth(bank, train_structures, seed=seed)
    method = METHODS[method_name](
        env, config_file=os.path.join(folder, "config.yaml"), enjoy_criteria=args.enjoy_criteria,
    )

    generator = torch.Generator().manual_seed(seed)
    expert_loader = DataLoader(data.train, batch_size=args.batch_size, shuffle=True, generator=generator)
    idm_loader = None
    if data.random is not None:
        idm_loader = DataLoader(data.random, batch_size=args.batch_size, shuffle=True, generator=generator)
    method.attach(TrainingContext(
        env=env, bank=bank, expert_loader=expert_loader, idm_loader=idm_loader,
        n_epochs=args.epochs, batch_size=args.batch_size, device=args.device,
        rollout_episodes=n_debug or len(train_structures), seed=seed,
    ))

    ckpt_path = os.path.join(folder, "checkpoint.pt")
    final_path = os.path.join(folder, "final.pt")
    metrics_path = os.path.join(folder, "train_metrics.jsonl")
    start_epoch, train_seconds = 0, 0.0
    if args.resume and os.path.exists(final_path):
        state = torch.load(final_path, map_location="cpu", weights_only=False)
        bank.load_state_dict(state["bank"])
        method.load_checkpoint_state(state["method"])
        start_epoch, train_seconds = args.epochs, state.get("train_seconds", 0.0)
        logger.info("Loaded final model, skipping training.")
    elif args.resume and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if state["epochs"] != args.epochs:
            raise RuntimeError(f"Checkpoint was made for {state['epochs']} epochs, not {args.epochs}.")
        bank.load_state_dict(state["bank"])
        method.load_checkpoint_state(state["method"])
        env.load_state_dict(state["env"])
        load_rng_state(state["rng"], generator)
        start_epoch, train_seconds = state["epoch_completed"], state["train_seconds"]
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as _file:
                kept = [line for line in _file if json.loads(line)["epoch"] < start_epoch]
            with open(metrics_path, "w", encoding="utf-8") as _file:
                _file.writelines(kept)
        logger.info("Resumed from epoch %d", start_epoch)
    else:
        for path in (ckpt_path, final_path, metrics_path):
            if os.path.exists(path):
                os.remove(path)

    # ---- training ----------------------------------------------------------
    log_every = max(1, args.epochs // 50)
    method.set_training(True)
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        metrics = method.run_epoch(epoch)
        train_seconds += time.time() - epoch_start
        metrics = {key: float(value) for key, value in metrics.items() if np.isscalar(value)}
        with open(metrics_path, "a", encoding="utf-8") as _file:
            _file.write(json.dumps({"epoch": epoch, "seconds": time.time() - epoch_start, **metrics}) + "\n")
        if epoch % log_every == 0 or epoch == args.epochs - 1:
            shown = ", ".join(f"{k}={v:.4g}" for k, v in list(metrics.items())[:6])
            logger.info("epoch %d/%d (%.1fs) %s", epoch + 1, args.epochs, time.time() - epoch_start, shown)
        if (epoch + 1) % args.ckpt_every == 0 and epoch + 1 < args.epochs:
            save_checkpoint(ckpt_path, {
                "epochs": args.epochs, "epoch_completed": epoch + 1, "train_seconds": train_seconds,
                "method": method.checkpoint_state(), "bank": bank.state_dict(),
                "env": env.state_dict(), "rng": rng_state(generator),
            })
    method.set_training(False)
    if start_epoch < args.epochs:
        save_checkpoint(final_path, {
            "epochs": args.epochs, "train_seconds": train_seconds,
            "method": method.checkpoint_state(), "bank": bank.state_dict(),
        })
    logger.info("Training done: %d epochs, %.1f min, %d guarded resets",
                args.epochs, train_seconds / 60, env.resets)

    # ---- evaluation (separate environment, built after training) ----------
    eval_started = time.time()
    eval_env = LabyrinthImageEnv(bank)
    model = returns_model(eval_env)
    split_structures = {
        name: data.splits[name].structures[:n_debug] if n_debug else data.splits[name].structures
        for name in ("train", "eval", "test")
    }
    results_splits, visited = {}, []
    for name, structures in split_structures.items():
        split_results, split_visited = evaluate(
            method, eval_env, structures, name, seed, model, record=name in ("eval", "test"),
        )
        results_splits[name] = split_results
        visited += split_visited
        logger.info("%s: AER %.2f +/- %.2f (raw %.3f) SR %.2f%%", name, split_results["aer"],
                    split_results["aer_std"], split_results["aer_raw"], split_results["sr"])
    if args.extra_splits:
        for name, mode in EXTRA_SPLITS.items():
            split_results, _ = evaluate(
                method, eval_env, split_structures["train"], name, seed, model, mode=mode,
            )
            results_splits[name] = split_results
            logger.info("%s: AER %.2f +/- %.2f SR %.2f%%", name, split_results["aer"],
                        split_results["aer_std"], split_results["sr"])

    # ---- encoding analysis ---------------------------------------------------
    train_info = train_state_info(data.splits["train"], data.train, eval_env)
    frame, encoding_summary = encoding_analysis(
        method, bank, train_info, train_structures, visited,
        {name: split_structures[name] for name in ("eval", "test")},
    )
    analysis_path = save_frame(frame, os.path.join(folder, "encoding_analysis"))
    for name, summary in encoding_summary.items():
        logger.info("encoding %s: %s", name, json.dumps(summary))

    hyper = dict(method.hyperparameters)
    results = {
        "config": {
            "method": method_name,
            "method_class": f"{type(method).__module__}.{type(method).__name__}",
            "encoder": encoder,
            "epochs": args.epochs,
            "lr": hyper.get("lr"),
            "idm_lr": hyper.get("idm_lr") if method_name in IDM_METHODS else None,
            "batch_size": args.batch_size,
            "seed": seed,
            "device": args.device,
            "hyperparameters": hyper,
            "enjoy_criteria": args.enjoy_criteria if method_name in IDM_METHODS else None,
            "debug_n_labyrinths": n_debug,
            "il_datasets": il_info,
            "il_datasets_commit": il_info["commit"],
            "dataset": {"repo": HF_REPO, "revision": HF_REVISION},
            "dataset_revision": HF_REVISION,
            "image_size": [64, 64, 3],
            "model_selection": "final checkpoint after the last epoch",
            "returns_model": model,
            "aer_normalisation": "per labyrinth: -4 + (R + 4) * 5 / (R_shortest + 4); "
                                 "failure -> -4.00, shortest path -> 1.00",
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "splits": results_splits,
        "encoding_analysis": encoding_summary,
        "encoding_analysis_file": os.path.basename(analysis_path),
        "training": {
            "guarded_resets": env.resets,
            "dagger_rounds": getattr(method, "collected", None),
            "env_steps": getattr(method, "env_steps", None),
            "gradient_steps": getattr(method, "gradient_steps", None),
            "idm_dataset_size": len(idm_loader.dataset) if idm_loader is not None else None,
        },
        "timing": {
            "train_s": train_seconds,
            "eval_s": time.time() - eval_started,
            "total_s": time.time() - started,
        },
    }
    write_json(os.path.join(folder, "results.json"), results)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    logger.info("Wrote %s", os.path.join(folder, "results.json"))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def child_command(args: argparse.Namespace, method: str, encoder: str, seed: int) -> List[str]:
    cmd = [
        sys.executable, os.path.abspath(__file__), "--run-one", method, encoder, str(seed),
        "--epochs", str(args.epochs), "--device", args.device, "--out", args.out,
        "--batch-size", str(args.batch_size), "--ckpt-every", str(args.ckpt_every),
        "--enjoy-criteria", str(args.enjoy_criteria),
    ]
    if args.resume:
        cmd.append("--resume")
    if args.extra_splits:
        cmd.append("--extra-splits")
    if args.debug_n_labyrinths:
        cmd += ["--debug-n-labyrinths", str(args.debug_n_labyrinths)]
    if args.data_dir:
        cmd += ["--data-dir", args.data_dir]
    return cmd


def main() -> None:
    args = get_args()
    if args.run_one:
        method, encoder, seed = args.run_one
        try:
            run_single(args, method, encoder, int(seed))
        except Exception:  # pylint: disable=broad-except
            logger.error("Run failed:\n%s", traceback.format_exc())
            sys.exit(1)
        return

    os.makedirs(args.out, exist_ok=True)
    setup_logging(os.path.join(args.out, "ablation.log"))
    il_info = il_datasets_info()
    logger.info("IL-Datasets %s @ %s (%s)", il_info["version"], il_info["commit"], il_info["requested_revision"])
    logger.info("Dataset %s @ %s", HF_REPO, HF_REVISION)
    report = preflight(args, args.out)
    logger.info("Preflight passed in %.0fs: %s", report["seconds"], json.dumps(report["splits"]))

    failures = []
    for seed in args.seeds:
        for method in args.methods:
            for encoder in args.encoders:
                folder = run_dir(args.out, method, encoder, seed)
                if args.resume and os.path.exists(os.path.join(folder, "results.json")):
                    logger.info("Skipping %s/%s/seed_%d (results.json exists)", method, encoder, seed)
                    continue
                os.makedirs(folder, exist_ok=True)
                logger.info("Starting %s/%s/seed_%d", method, encoder, seed)
                run_started = time.time()
                cmd, env = child_command(args, method, encoder, seed), None
                if sys.platform == "win32" and sys.prefix != sys.base_prefix:
                    # Launch the base interpreter directly instead of the venv's python.exe
                    # trampoline (which Smart App Control may block); __PYVENV_LAUNCHER__
                    # makes it run inside the venv, same as multiprocessing does on Windows.
                    env = dict(os.environ, __PYVENV_LAUNCHER__=sys.executable)
                    cmd[0] = getattr(sys, "_base_executable", sys.executable)
                with open(os.path.join(folder, "console.log"), "a", encoding="utf-8") as console:
                    code = subprocess.call(cmd, stdout=console, stderr=subprocess.STDOUT, cwd=ROOT, env=env)
                minutes = (time.time() - run_started) / 60
                if code == 0:
                    logger.info("Finished %s/%s/seed_%d in %.1f min", method, encoder, seed, minutes)
                else:
                    failures.append({"method": method, "encoder": encoder, "seed": seed, "exit_code": code})
                    logger.error("FAILED %s/%s/seed_%d (exit %d) after %.1f min; see %s",
                                 method, encoder, seed, code, minutes, os.path.join(folder, "log.txt"))

    write_json(os.path.join(args.out, "failures.json"), {"failures": failures})
    logger.info("Summary: %s", write_summary(args.out, args.methods, args.encoders, args.seeds))
    logger.info("Table: %s", write_table(args.out, args.methods, args.encoders, args.seeds))
    if failures:
        logger.error("%d run(s) failed: %s", len(failures), failures)


if __name__ == "__main__":
    main()
