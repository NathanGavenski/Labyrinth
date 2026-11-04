"""IL-Datasets methods (main branch) adapted to image Labyrinth without modifying them.

What is reused unchanged:
    BC    ``BC._train``
    BCO   ``BCO._train`` and ``BCO._append_samples`` (policy rollouts appended to the
          IDM data at epoch 0 and every ``enjoy_criteria`` epochs, as in ``BCO.train``)
    IUPE  ``IUPE._train``, ``IUPE.predict`` (sampling while training) and
          ``ABCO._append_samples``
    GAIL  ``GAIL._rollout``, ``GAIL._train_discriminator``, ``GAIL._train`` (PPO + GAE)

What is replaced, because main does not support image observations here:
    * ``_enjoy``: main calls ``predict`` without transforms (raises for visual
      policies) and IUPE/ABCO call ``reached_goal`` (no Labyrinth entry). The
      replacement runs the same 100-episode loop on the guarded train environment.
    * BCO/IUPE IDM encoder: main concatenates (s, s') into 6 channels but builds a
      3-channel encoder. It is rebuilt with 6 input channels from the same IL-Datasets
      class; the IDM head (IL-Datasets ``MLP``) is kept.
    * GAIL: main's critic, discriminator and running normaliser are MLPs over flat
      vectors. The discriminator is the fixed small CNN below (same in both encoder
      conditions), the critic uses the same small CNN trunk, and pixels are scaled
      to [0, 1] instead of running-normalised.
"""
from typing import Any, Dict

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from imitation_datasets.baselines.methods import BC, BCO, IUPE, GAIL
from imitation_datasets.baselines.methods.method import Method
from imitation_datasets.baselines.methods.policies import CNN, Resnet
from imitation_datasets.baselines.methods.utils import import_hyperparameters

from .base import LabyrinthMixin, Scale


# ---------------------------------------------------------------------------
# BC
# ---------------------------------------------------------------------------
class LabyrinthBC(LabyrinthMixin, BC):
    """IL-Datasets BC; one epoch is one pass of ``BC._train`` over the train split."""

    def __init__(self, environment, config_file: str, enjoy_criteria: int = 100) -> None:
        BC.__init__(self, environment, enjoy_criteria=enjoy_criteria, config_file=config_file)
        self.prepare_visual_policy()

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        return self._train(self.ctx.expert_loader)


# ---------------------------------------------------------------------------
# BCO / IUPE
# ---------------------------------------------------------------------------
def rebuild_idm(method: BCO) -> None:
    """Replace the IDM encoder with a 6-channel one of the configured type."""
    idm = method.hyperparameters.get("idm", "CnnPolicy")
    height, width, channels = method.observation_size
    shape = (height, width, channels * 2)
    if idm == "CnnPolicy":
        encoder = CNN(shape)
    elif idm == "ResnetPolicy":
        encoder = Resnet(shape)
    else:
        raise ValueError(f"Unsupported IDM encoder {idm}")
    head = method.idm[1]
    with torch.no_grad():
        out = encoder(torch.zeros(1, channels * 2, height, width))
    if out.shape[-1] != head.layers[0].in_features:
        raise ValueError("IDM head does not match the 6-channel encoder output.")
    method.idm = nn.Sequential(Scale(), encoder, head)
    method.idm_optimizer = optim.Adam(method.idm.parameters(), lr=method.hyperparameters["idm_lr"])


class _IdmMixin(LabyrinthMixin):
    """Shared epoch loop for BCO and IUPE, mirroring ``BCO.train``."""

    def _loaders(self) -> Dict[str, Any]:
        return {"idm_dataset": self.ctx.idm_loader, "expert_dataset": self.ctx.expert_loader}

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        loaders = self._loaders()
        metrics = self._train(**loaders)
        if epoch % self.enjoy_criteria == 0:
            before = len(self.ctx.idm_loader.dataset)
            self.set_training(True)
            self._append_samples(loaders)
            metrics["idm_dataset_size"] = len(self.ctx.idm_loader.dataset)
            metrics["idm_dataset_before"] = before
        return metrics

    def extra_state(self) -> Dict[str, Any]:
        return {"idm_dataset": self.ctx.idm_loader.dataset.state_dict()}

    def load_extra_state(self, state: Dict[str, Any]) -> None:
        if "idm_dataset" in state:
            self.ctx.idm_loader.dataset.load_state_dict(state["idm_dataset"])


class LabyrinthBCO(_IdmMixin, BCO):
    """IL-Datasets BCO (with its I_pos appends) on image Labyrinth."""

    def __init__(self, environment, config_file: str, enjoy_criteria: int = 100) -> None:
        BCO.__init__(self, environment, enjoy_criteria=enjoy_criteria, config_file=config_file)
        self.prepare_visual_policy()
        rebuild_idm(self)


class LabyrinthIUPE(_IdmMixin, IUPE):
    """IL-Datasets IUPE (ABCO appends, sampled IDM labels and actions) on image Labyrinth."""

    def __init__(self, environment, config_file: str, enjoy_criteria: int = 100) -> None:
        IUPE.__init__(self, environment, enjoy_criteria=enjoy_criteria, config_file=config_file)
        self.prepare_visual_policy()
        rebuild_idm(self)

    def set_training(self, training: bool) -> None:
        self.is_training = training


# ---------------------------------------------------------------------------
# GAIL
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Fixed GAIL trunk: Conv(3->32,k8,s4)-LReLU-Conv(32->64,k4,s2)-LReLU-Conv(64->64,k3,s1)-LReLU."""

    def __init__(self, channels: int = 3, size: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            Scale(),
            nn.Conv2d(channels, 32, kernel_size=8, stride=4), nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.LeakyReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.layers(torch.zeros(1, channels, size, size)).shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Discriminator(nn.Module):
    """D(s, a): SmallCNN features + one-hot action -> FC 256 -> FC 256 -> FC 1 (logit)."""

    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.trunk = SmallCNN()
        self.head = nn.Sequential(
            nn.Linear(self.trunk.out_dim + action_size, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.trunk(state), action], dim=1))


class Critic(nn.Module):
    """V(s): SmallCNN features -> FC 256 -> FC 256 -> FC 1."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = SmallCNN()
        self.head = nn.Sequential(
            nn.Linear(self.trunk.out_dim, 256), nn.LeakyReLU(),
            nn.Linear(256, 256), nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(state))


class DeviceBridge(nn.Module):
    """Runs a module on its own device and returns CPU outputs.

    GAIL's rollout, GAE and PPO code works on CPU tensors; this keeps that code
    unchanged while the networks run on the GPU.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        device = next(self.module.parameters()).device
        return self.module(*[arg.to(device) for arg in args]).cpu()


class PixelIdentity(nn.Module):
    """Stands in for GAIL's RunningNormalizer; scaling happens inside each network."""

    def update(self, x: torch.Tensor) -> None:  # pylint: disable=unused-argument
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


class LabyrinthGAIL(LabyrinthMixin, GAIL):
    """IL-Datasets GAIL (PPO generator) with image networks.

    One epoch is one ``GAIL`` iteration: ``n_steps`` environment steps, one
    discriminator pass and ``ppo_epochs`` PPO passes over the rollout.
    """

    checkpoint_modules = LabyrinthMixin.checkpoint_modules

    def __init__(self, environment, config_file: str, enjoy_criteria: int = 100) -> None:
        # GAIL.__init__ builds MLP critic/discriminator from the observation size,
        # which fails for image shapes, so its attribute setup is reproduced here.
        self.enjoy_criteria = enjoy_criteria
        self.verbose = False
        self.environment_name = environment.spec.name
        self.save_path = f"./tmp/gail/{self.environment_name}/"
        self.hyperparameters = import_hyperparameters(config_file, self.environment_name)
        Method.__init__(self, environment, self.hyperparameters)
        if not self.discrete:
            raise ValueError("Labyrinth has discrete actions.")
        self.prepare_visual_policy()

        self.normalizer = PixelIdentity()
        self.critic = DeviceBridge(Critic())
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.hyperparameters["critic_lr"]
        )
        self.discriminator = DeviceBridge(Discriminator(self.action_size))
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.hyperparameters["discriminator_lr"]
        )
        self.log_std = None
        self._obs = None
        self._accumulated_reward = 0.0
        self._expert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Policy logits on the policy's device, returned on the CPU like GAIL expects."""
        device = next(self.policy.parameters()).device
        return self.policy(self.normalizer(x).to(device)).cpu()

    def discriminator_logits(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = F.one_hot(action.long().view(-1), self.action_size).float()
        return self.discriminator(self.normalizer(state), action).squeeze(1)

    def run_epoch(self, epoch: int) -> Dict[str, float]:
        if self._expert is None:
            self._expert = self._expert_sampler(self.ctx.expert_loader)
        if self._obs is None:
            self._obs, _ = self.environment.reset()
            self._accumulated_reward = 0.0
        rollout, self._obs, episode_rewards = self._rollout(self._obs)
        metrics = {**self._train_discriminator(rollout, self._expert), **self._train(rollout)}
        if episode_rewards:
            metrics["rollout_reward"] = float(np.mean(episode_rewards))
        metrics["rollout_episodes"] = len(episode_rewards)
        return metrics
