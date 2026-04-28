"""Compatibility package exposing `AlgoTune/AlgoTuneTasks` from repo root."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / "AlgoTune" / "AlgoTuneTasks"
__path__ = [str(_ROOT)]
