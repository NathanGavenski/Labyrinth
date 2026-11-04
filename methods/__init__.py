"""Imitation learning methods for the Labyrinth encoder ablation.

DAgger and SQIL are implemented here as IL-Datasets ``Method`` subclasses. BC, BCO,
IUPE and GAIL come from IL-Datasets (main) through thin visual adapters.
"""
import ablation  # noqa: F401  (Labyrinth import path and platform shims)

from .adapters import LabyrinthBC, LabyrinthBCO, LabyrinthIUPE, LabyrinthGAIL
from .dagger import DAgger
from .sqil import SQIL

METHODS = {
    "bc": LabyrinthBC,
    "bco": LabyrinthBCO,
    "iupe": LabyrinthIUPE,
    "gail": LabyrinthGAIL,
    "dagger": DAgger,
    "sqil": SQIL,
}
