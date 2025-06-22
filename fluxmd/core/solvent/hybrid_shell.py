"""Hybrid explicit solvent shell utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class HybridSolventShell:
    """Representation of a spherical solvent shell."""

    water_coords: np.ndarray
    center: np.ndarray
    inner_radius: float
    outer_radius: float

    def count_waters(self) -> int:
        """Return number of waters in the shell."""
        return water_count_in_shell(
            self.water_coords, self.center, self.inner_radius, self.outer_radius
        )


def water_count_in_shell(
    water_coords: Iterable[np.ndarray] | np.ndarray,
    center: Iterable[float] | np.ndarray,
    inner_radius: float,
    outer_radius: float,
) -> int:
    """Count water molecules within a radial shell.

    Args:
        water_coords: Coordinates of water oxygens with shape (N, 3).
        center: Center of the shell.
        inner_radius: Inner radius in Å.
        outer_radius: Outer radius in Å.

    Returns:
        Number of waters whose distance from ``center`` is >= ``inner_radius`` and
        <= ``outer_radius``.
    """
    coords = np.asarray(water_coords, dtype=float)
    c = np.asarray(center, dtype=float)
    dists = np.linalg.norm(coords - c, axis=1)
    mask = (dists >= inner_radius) & (dists <= outer_radius)
    return int(mask.sum())
