"""Support package for the encoder ablation benchmark (benchmark_ablation.py).

Importing this package makes the Labyrinth environment importable from the repo
checkout (``src/`` on ``sys.path``), runs pygame headless, and installs a no-op
``resource`` module on platforms without one (Windows). The environment only uses
``resource`` to raise the stack limit for deep DFS recursion, which 5x5 labyrinths
do not need. On Linux the real module is used and nothing changes.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import resource  # noqa: F401  pylint: disable=unused-import
except ImportError:
    _resource = types.ModuleType("resource")
    _resource.RLIMIT_STACK = 0
    _resource.RLIM_INFINITY = -1
    _resource.getrlimit = lambda *_: (0, 0)
    _resource.setrlimit = lambda *_: None
    sys.modules["resource"] = _resource

import labyrinth  # noqa: E402,F401  pylint: disable=wrong-import-position  (registers Labyrinth-v0)
