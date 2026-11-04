"""DAgger (Ross, Gordon and Bagnell, 2011) for image Labyrinth.

The supervised step is IL-Datasets' ``BC._train`` (same policy classes, Adam and
lr). Training is split into ``rounds`` rounds of equal length so the total number of
gradient epochs equals the requested budget:

* Round 1 uses beta = 1, i.e. the expert's own trajectories, which are the train
  split itself, so it is BC on the dataset.
* Before round i (i >= 2) the mixture policy pi_i = beta_i * expert + (1 - beta_i) *
  policy (per-step choice, greedy policy action) runs one episode on each train
  configuration, with beta_i = beta_decay ** (i - 1). Every visited state is
  labelled with the expert action and appended to the aggregated dataset, and
  training continues from the current weights on all data collected so far.

The expert is the shortest-path solver of the environment from the agent's tile
(``GuardedLabyrinth.expert_action``), converted with ``create_expert.state_to_action``.
Rollouts only ever happen in the guarded train environment.
"""
import os
from typing import Any, Dict, List, Tuple

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch.utils.data import DataLoader

from imitation_datasets.baselines.methods import BC

from ablation.data import IndexedTransitions
from .base import LabyrinthMixin


PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = f"{PATH}/config/dagger.yaml"


def round_starts(n_epochs: int, rounds: int) -> List[int]:
    """First epoch of each round (rounds are capped by the number of epochs)."""
    rounds = max(1, min(rounds, n_epochs))
    return [int(round(i * n_epochs / rounds)) for i in range(rounds)]


def beta_schedule(rounds: int, decay: float) -> List[float]:
    """beta_1 = 1, beta_i = decay ** (i - 1)."""
    return [decay ** i for i in range(rounds)]


class DAgger(LabyrinthMixin, BC):
    """Dataset Aggregation with a shortest-path expert."""

    __version__ = "1.0.0"
    __author__ = "Ross et. al."
    __method_name__ = "Dataset Aggregation"

    def __init__(
        self,
        environment,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None,
    ) -> None:
        BC.__init__(
            self, environment, enjoy_criteria=enjoy_criteria, verbose=verbose,
            config_file=CONFIG_FILE if config_file is None else config_file,
        )
        self.save_path = f"./tmp/dagger/{self.environment_name}/"
        if self.visual:
            self.prepare_visual_policy()
        self.rounds = int(self.hyperparameters.get("rounds", 10))
        self.beta_decay = float(self.hyperparameters.get("beta_decay", 0.5))
        horizon = self.hyperparameters.get("rollout_horizon", None)
        self.rollout_horizon = None if horizon in (None, 0) else int(horizon)
        self.rng = np.random.default_rng(int(self.hyperparameters.get("seed", 0)))
        self.aggregated: IndexedTransitions = None
        self.loader: DataLoader = None
        self.collected: List[Dict[str, Any]] = []

    def attach(self, ctx) -> None:
        super().attach(ctx)
        self.rng = np.random.default_rng(ctx.seed)

    # -- data ------------------------------------------------------------------
    def _init_aggregate(self) -> None:
        base = self.ctx.expert_loader.dataset
        self.aggregated = IndexedTransitions(
            base.bank, base.states.clone(), base.actions.clone(),
            base.next_states.clone(), base.dones.clone(),
        )
        self.loader = DataLoader(
            self.aggregated, batch_size=self.ctx.batch_size, shuffle=True,
        )

    def collect(self, beta: float) -> Tuple[List[int], List[int], Dict[str, float]]:
        """One beta-mixture episode per train configuration, expert-labelled."""
        env = self.environment
        states, labels = [], []
        expert_steps, returns, successes = 0, [], []
        for _ in range(self.rollout_episodes):
            obs, _ = env.reset()
            done, steps, total, reached = False, 0, 0.0, False
            while not done and (self.rollout_horizon is None or steps < self.rollout_horizon):
                expert = env.expert_action()
                states.append(env.obs_index)
                labels.append(expert)
                if self.rng.random() < beta:
                    action = expert
                    expert_steps += 1
                else:
                    action = self.greedy_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total += reward
                reached |= bool(terminated)
                steps += 1
            returns.append(total)
            successes.append(reached)
        info = {
            "beta": beta,
            "states": len(states),
            "expert_fraction": expert_steps / max(1, len(states)),
            "mixture_success_rate": float(np.mean(successes)),
            "mixture_return": float(np.mean(returns)),
        }
        return states, labels, info

    def aggregate(self, states: List[int], labels: List[int]) -> None:
        states = torch.tensor(states, dtype=torch.long).view(-1, 1)
        labels = torch.tensor(labels, dtype=torch.long).view(-1, 1)
        data = self.aggregated
        data.states = torch.cat((data.states, states))
        data.next_states = torch.cat((data.next_states, states))  # unused by BC._train
        data.actions = torch.cat((data.actions, labels))
        data.dones = torch.cat((data.dones, torch.zeros(len(states), dtype=torch.bool)))

    # -- training ----------------------------------------------------------------
    def run_epoch(self, epoch: int) -> Dict[str, float]:
        if self.aggregated is None:
            self._init_aggregate()
        starts = round_starts(self.ctx.n_epochs, self.rounds)
        betas = beta_schedule(len(starts), self.beta_decay)
        metrics = {}
        if epoch in starts[1:]:
            round_idx = starts.index(epoch)
            states, labels, info = self.collect(betas[round_idx])
            self.aggregate(states, labels)
            info.update({"round": round_idx + 1, "epoch": epoch, "dataset_size": len(self.aggregated)})
            self.collected.append(info)
            metrics.update({f"dagger_{key}": value for key, value in info.items()})
        metrics.update(self._train(self.loader))
        metrics["dataset_size"] = len(self.aggregated)
        return metrics

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,
        always_save: bool = False,
        folder: str = None,
    ) -> Self:
        """Standalone training with the same interface as IL-Datasets' methods.

        ``self.environment`` must be a ``GuardedLabyrinth`` and ``train_dataset`` a
        DataLoader over ``IndexedTransitions``. No checkpoint selection is done.
        """
        from .base import TrainingContext  # pylint: disable=import-outside-toplevel

        if not hasattr(self, "ctx"):
            self.attach(TrainingContext(
                env=self.environment, bank=train_dataset.dataset.bank,
                expert_loader=train_dataset, idm_loader=None, n_epochs=n_epochs,
                batch_size=train_dataset.batch_size, device=self.device,
            ))
        for epoch in range(n_epochs):
            self.run_epoch(epoch)
        return self

    # -- checkpoints -------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        return {
            "aggregated": None if self.aggregated is None else self.aggregated.state_dict(),
            "rng": self.rng.bit_generator.state,
            "collected": self.collected,
        }

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        if state.get("aggregated") is not None:
            self._init_aggregate()
            self.aggregated.load_state_dict(state["aggregated"])
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]
        self.collected = state.get("collected", [])
