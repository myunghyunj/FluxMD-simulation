from __future__ import annotations

"""Central random number generation utilities.

Provides a simple factory for NumPy ``Generator`` instances so that all
components share consistent seeding behaviour.  Modules should request a
Generator explicitly rather than relying on ``np.random``'s global state.
"""

from typing import Optional

import numpy as np

__all__ = ["create_generator"]


def create_generator(seed: Optional[int] = None) -> np.random.Generator:
    """Return a new NumPy ``Generator`` with the given ``seed``.

    Parameters
    ----------
    seed:
        Seed for the generator.  ``None`` uses NumPy's default seeding
        mechanism.
    """

    return np.random.default_rng(seed)
