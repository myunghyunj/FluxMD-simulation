"""
Brownian-Langevin dynamics for molecular trajectory generation.

This module implements:
- BAOAB Langevin integrator
- Surface-constrained Brownian motion
- Monte Carlo layer hopping
- Geodesic guidance forces
"""

from .brownian_roller import BrownianSurfaceRoller, quaternion_multiply

__all__ = [
    "BrownianSurfaceRoller",
    "quaternion_multiply",
]
