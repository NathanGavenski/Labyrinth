"""SQIL (Reddy, Dragan and Levine, 2019) for image Labyrinth.

Soft Q-learning where expert transitions (train split) get reward 1 and the agent's
own transitions get reward 0. Every gradient step samples half of the batch from
each replay buffer. The Q-network is the IL-Datasets visual policy (same encoder
and head, 4 outputs), trained with Adam at the configured lr.

    Q-target   y = r + gamma * (1 - done) * alpha * logsumexp(Q_target(s') / alpha)
    loss       mean squared soft Bellman error
    acting     a ~ softmax(Q(s) / alpha) while training, argmax at evaluation
    target     Polyak averaging with ``tau`` after every gradient step

One epoch is one pass over the expert transitions: ceil(N_expert / (batch_size / 2))
gradient steps, each preceded by one environment step in the guarded train
environment. Before the first epoch, ``warmup_steps`` environment steps fill the
agent buffer. ``done`` is only set when the goal is reached (not on the step cap).
"""
import copy
import math
import os
from typing import Any, Dict

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from imitation_datasets.baselines.methods.method import Method
from imitation_datasets.baselines.methods.utils import import_hyperparameters

from .base import LabyrinthMixin


PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = f"{PATH}/config/sqil.yaml"


class ReplayBuffer:
    """Ring buffer of (state, action, next_state, done) with ImageBank indices."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.states = np.zeros(self.capacity, dtype=np.int64)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.next_states = np.zeros(self.capacity, dtype=np.int64)
        self.dones = np.zeros(self.capacity, dtype=bool)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state: int, action: int, next_state: int, done: bool) -> None:
        idx = self.position
        self.states[idx], self.actions[idx] = state, action
        self.next_states[idx], self.dones[idx] = next_state, done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @classmethod
    def from_transitions(cls, dataset) -> "ReplayBuffer":
        """Buffer holding all transitions of an IndexedTransitions dataset."""
        buffer = cls(len(dataset))
        for state, action, next_state, done in zip(
            dataset.states.view(-1).tolist(), dataset.actions.view(-1).tolist(),
            dataset.next_states.view(-1).tolist(), dataset.dones.view(-1).tolist(),
        ):
            buffer.add(state, action, next_state, done)
        return buffer

    def sample(self, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        idx = rng.integers(0, self.size, size=n)
        return {
            "states": self.states[idx], "actions": self.actions[idx],
            "next_states": self.next_states[idx], "dones": self.dones[idx],
        }

    def state_dict(self) -> Dict[str, Any]:
        size = self.size
        return {
            "capacity": self.capacity, "position": self.position, "size": size,
            "states": self.states[:size].copy(), "actions": self.actions[:size].copy(),
            "next_states": self.next_states[:size].copy(), "dones": self.dones[:size].copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.__init__(state["capacity"])
        size = state["size"]
        self.states[:size], self.actions[:size] = state["states"], state["actions"]
        self.next_states[:size], self.dones[:size] = state["next_states"], state["dones"]
        self.position, self.size = state["position"], size


def sqil_batch(
    expert: ReplayBuffer,
    agent: ReplayBuffer,
    batch_size: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Half expert transitions with reward 1, half agent transitions with reward 0."""
    half = batch_size // 2
    expert_batch = expert.sample(half, rng)
    agent_batch = agent.sample(batch_size - half, rng)
    batch = {key: np.concatenate([expert_batch[key], agent_batch[key]]) for key in expert_batch}
    batch["rewards"] = np.concatenate([
        np.ones(half, dtype=np.float32), np.zeros(batch_size - half, dtype=np.float32)
    ])
    batch["is_expert"] = batch["rewards"] == 1
    return batch


class SQIL(LabyrinthMixin, Method):
    """Soft Q Imitation Learning."""

    __version__ = "1.0.0"
    __author__ = "Reddy et. al."
    __method_name__ = "Soft Q Imitation Learning"

    def __init__(
        self,
        environment,
        enjoy_criteria: int = 100,
        verbose: bool = False,
        config_file: str = None,
    ) -> None:
        self.enjoy_criteria = enjoy_criteria
        self.verbose = verbose
        self.environment_name = environment.spec.name
        self.save_path = f"./tmp/sqil/{self.environment_name}/"
        self.hyperparameters = import_hyperparameters(
            CONFIG_FILE if config_file is None else config_file, self.environment_name,
        )
        super().__init__(environment, self.hyperparameters)
        if self.visual:
            self.prepare_visual_policy()
        self.target = copy.deepcopy(self.policy)
        self.target.requires_grad_(False)

        self.gamma = float(self.hyperparameters.get("gamma", 0.99))
        self.alpha = float(self.hyperparameters.get("alpha", 1.0))
        self.tau = float(self.hyperparameters.get("tau", 0.005))
        self.batch_size = int(self.hyperparameters.get("batch_size", 256))
        self.warmup_steps = int(self.hyperparameters.get("warmup_steps", 1000))
        self.agent_buffer = ReplayBuffer(int(self.hyperparameters.get("buffer_size", 1_000_000)))
        self.expert_buffer: ReplayBuffer = None
        self.rng = np.random.default_rng(int(self.hyperparameters.get("seed", 0)))
        self._obs = None
        self.env_steps = 0
        self.gradient_steps = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(x)

    def attach(self, ctx) -> None:
        super().attach(ctx)
        self.rng = np.random.default_rng(ctx.seed)
        if self.ctx.batch_size and "batch_size" not in self.hyperparameters:
            self.batch_size = ctx.batch_size
        self.expert_buffer = ReplayBuffer.from_transitions(ctx.expert_loader.dataset)

    # -- soft Q ------------------------------------------------------------------
    def soft_value(self, q_values: torch.Tensor) -> torch.Tensor:
        return self.alpha * torch.logsumexp(q_values / self.alpha, dim=1)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """Sample from the Boltzmann policy softmax(Q / alpha)."""
        self.policy.eval()
        q_values = self.logits(torch.as_tensor(obs)[None])[0]
        probs = torch.softmax(q_values / self.alpha, dim=0).double().cpu().numpy()
        return int(self.rng.choice(len(probs), p=probs / probs.sum()))

    def env_step(self) -> None:
        env = self.environment
        if self._obs is None:
            self._obs, _ = env.reset()
        state = env.obs_index
        action = self.act(self._obs)
        self._obs, _, terminated, truncated, _ = env.step(action)
        self.agent_buffer.add(state, action, env.obs_index, bool(terminated))
        self.env_steps += 1
        if terminated or truncated:
            self._obs, _ = env.reset()

    def _train(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        images = self.ctx.bank.tensor()
        device = self.device
        states = images[torch.from_numpy(batch["states"])].to(device)
        next_states = images[torch.from_numpy(batch["next_states"])].to(device)
        actions = torch.from_numpy(batch["actions"]).to(device)
        dones = torch.from_numpy(batch["dones"]).float().to(device)
        rewards = torch.from_numpy(batch["rewards"]).to(device)

        self.policy.train()
        self.target.eval()
        with torch.no_grad():
            target = rewards + self.gamma * (1 - dones) * self.soft_value(self.target(next_states))
        q_values = self.policy(states)
        q_taken = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
        loss = F.mse_loss(q_taken, target)

        self.optimizer_fn.zero_grad()
        loss.backward()
        self.optimizer_fn.step()
        self._update_target()
        self.gradient_steps += 1

        expert = torch.from_numpy(batch["is_expert"]).to(device)
        return {
            "loss": loss.item(),
            "q_expert": q_taken[expert].mean().item(),
            "q_agent": q_taken[~expert].mean().item(),
            "accuracy_expert": (q_values[expert].argmax(1) == actions[expert]).float().mean().item() * 100,
        }

    @torch.no_grad()
    def _update_target(self) -> None:
        for target, online in zip(self.target.parameters(), self.policy.parameters()):
            target.mul_(1 - self.tau).add_(online, alpha=self.tau)
        for target, online in zip(self.target.buffers(), self.policy.buffers()):
            target.copy_(online)

    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.expert_buffer) / (self.batch_size // 2))

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        while self.env_steps < self.warmup_steps:
            self.env_step()
        accumulated = []
        for _ in range(self.steps_per_epoch()):
            self.env_step()
            accumulated.append(self._train(sqil_batch(
                self.expert_buffer, self.agent_buffer, self.batch_size, self.rng,
            )))
        metrics = {key: float(np.nanmean([m[key] for m in accumulated])) for key in accumulated[0]}
        metrics.update({"env_steps": self.env_steps, "gradient_steps": self.gradient_steps,
                        "agent_buffer": len(self.agent_buffer)})
        return metrics

    def train(
        self,
        n_epochs: int,
        train_dataset: DataLoader,
        eval_dataset: DataLoader = None,
        always_save: bool = False,
        folder: str = None,
    ) -> Self:
        """Standalone training with the same interface as IL-Datasets' methods."""
        from .base import TrainingContext  # pylint: disable=import-outside-toplevel

        if not hasattr(self, "ctx"):
            self.attach(TrainingContext(
                env=self.environment, bank=train_dataset.dataset.bank,
                expert_loader=train_dataset, idm_loader=None, n_epochs=n_epochs,
                batch_size=self.batch_size, device=self.device,
            ))
        for epoch in range(n_epochs):
            self.run_epoch(epoch)
        return self

    def save(self, path: str = None, name: str = None) -> None:
        path = self.save_path if path is None else path
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), f"{path}/{'best_model' if name is None else name}.ckpt")

    def load(self, path: str = None, name: str = None) -> Self:
        path = self.save_path if path is None else path
        name = "best_model" if name is None else name
        self.policy.load_state_dict(torch.load(f"{path}/{name}.ckpt", map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())
        return self

    # -- checkpoints -------------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        return {
            "agent_buffer": self.agent_buffer.state_dict(),
            "rng": self.rng.bit_generator.state,
            "env_steps": self.env_steps,
            "gradient_steps": self.gradient_steps,
        }

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        if "agent_buffer" in state:
            self.agent_buffer.load_state_dict(state["agent_buffer"])
            self.rng.bit_generator.state = state["rng"]
            self.env_steps = state["env_steps"]
            self.gradient_steps = state["gradient_steps"]
