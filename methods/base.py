"""Shared plumbing for running IL-Datasets methods on image Labyrinth.

All observations are uint8 3x64x64 arrays (ImageBank). Policies get a parameter-free
``Scale`` module in front of the encoder, which divides by 255 like ``ToTensor``.
The encoders themselves (``CNN``/``Resnet``) and heads are the ones IL-Datasets'
``Method`` builds from the YAML config.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from imitation_datasets.dataset.metrics import average_episodic_reward


class Scale(nn.Module):
    """uint8/[0, 255] pixels -> float in [0, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() / 255.0


@dataclass
class TrainingContext:
    """Everything a method needs for training on the guarded train environment."""
    env: Any                                # ablation.envs.GuardedLabyrinth
    bank: Any                               # ablation.data.ImageBank
    expert_loader: DataLoader               # train split transitions
    idm_loader: Optional[DataLoader]        # random split transitions (BCO/IUPE)
    n_epochs: int
    batch_size: int
    device: str
    rollout_episodes: int = 100             # episodes per training-time rollout phase
    seed: int = 0


class LabyrinthMixin:
    """Epoch-level interface used by benchmark_ablation.py for every method.

    Subclasses implement ``run_epoch(epoch)``. ``train()`` of the IL-Datasets methods
    is not used because it cannot resume and selects checkpoints with ``_enjoy``.
    """

    checkpoint_modules = (
        "policy", "optimizer_fn", "idm", "idm_optimizer", "critic", "critic_optimizer",
        "discriminator", "discriminator_optimizer", "target",
    )

    def prepare_visual_policy(self) -> None:
        """Prefix the encoder with pixel scaling; parameters (and the optimizer) are unchanged."""
        encoder, head = self.policy[0], self.policy[1]
        if not isinstance(encoder, nn.Sequential) or not isinstance(encoder[0], Scale):
            self.policy[0] = nn.Sequential(Scale(), encoder)
        if not isinstance(head, nn.Sequential) or not isinstance(head[-1], nn.Linear):
            raise TypeError("Expected the IL-Datasets visual policy (encoder, Sequential head).")

    def attach(self, ctx: TrainingContext) -> None:
        """Bind the training context and move models to the device."""
        self.ctx = ctx
        self.environment = ctx.env
        self.rollout_episodes = ctx.rollout_episodes
        self.device = ctx.device
        for name in self.checkpoint_modules:
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                module.to(self.device)

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        raise NotImplementedError()

    def set_training(self, training: bool) -> None:
        """Hook for methods that act differently while training (IUPE)."""

    # -- checkpoints -----------------------------------------------------------
    def extra_state(self) -> Dict[str, Any]:
        return {}

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        pass

    def checkpoint_state(self) -> Dict[str, Any]:
        state = {}
        for name in self.checkpoint_modules:
            module = getattr(self, name, None)
            if module is not None and hasattr(module, "state_dict"):
                state[name] = module.state_dict()
        state["extra"] = self.extra_state()
        return state

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        for name in self.checkpoint_modules:
            module = getattr(self, name, None)
            if module is not None and name in state:
                module.load_state_dict(state[name])
        self.load_extra_state(state.get("extra", {}))

    # -- inference -------------------------------------------------------------
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Action logits (Q-values for SQIL) for a uint8 batch."""
        return self.forward(x.to(self.device))

    @torch.no_grad()
    def greedy_action(self, obs: np.ndarray) -> int:
        self.policy.eval()
        return int(self.logits(torch.as_tensor(obs)[None]).argmax(1).item())

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer features: the input of the final linear layer."""
        self.policy.eval()
        x = x.to(next(self.policy.parameters()).device)
        return self.policy[1][:-1](self.policy[0](x))

    # -- rollouts on the guarded train environment ----------------------------
    def labyrinth_rollouts(self, n_episodes: int, act, return_ipos: bool = False):
        """Run ``n_episodes`` on ``self.environment`` with ``act(obs) -> action``.

        Returns metrics (aer, aer_std, success_rate) and, optionally, the visited
        transitions as ImageBank indices in IL-Datasets' I_pos format.
        """
        env = self.environment
        returns, successes = [], []
        i_pos = defaultdict(list)
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done, reached, total = False, False, 0.0
            while not done:
                state_index = env.obs_index
                action = int(act(obs))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                done = terminated or truncated
                reached |= bool(terminated)
                if return_ipos:
                    i_pos["states"].append([state_index])
                    i_pos["actions"].append(action)
                    i_pos["next_states"].append([env.obs_index])
                    i_pos["dones"].append(done)
            returns.append(total)
            successes.append(reached)

        metrics = average_episodic_reward(returns)
        metrics["success_rate"] = float(np.mean(successes))
        if not return_ipos:
            return metrics
        i_pos = {
            "states": np.array(i_pos["states"], dtype=np.int64).reshape(-1, 1),
            "actions": np.array(i_pos["actions"], dtype=np.int64),
            "next_states": np.array(i_pos["next_states"], dtype=np.int64).reshape(-1, 1),
            "dones": np.array(i_pos["dones"], dtype=bool),
        }
        return metrics, i_pos

    def _enjoy(self, render=False, teacher_reward=None, random_reward=None,
               return_ipos=False, **_):
        """Visual replacement for IL-Datasets' ``_enjoy`` (runs on the guarded env)."""
        def act(obs):
            return self.predict(torch.as_tensor(obs)[None])
        return self.labyrinth_rollouts(self.rollout_episodes, act, return_ipos)
