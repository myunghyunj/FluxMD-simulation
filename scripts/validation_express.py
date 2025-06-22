#!/usr/bin/env python3
"""Quick validation for cylindrical sampling coverage."""

from __future__ import annotations

import numpy as np

np.random.seed(0)

from fluxmd.core.cylindrical_sampler import FastCylindricalSampler


def coverage_cv(points: np.ndarray, n_theta: int = 18, n_z: int = 20) -> float:
    """Return coefficient of variation of hits on cylindrical bins."""
    theta_bins = np.linspace(-np.pi, np.pi, n_theta + 1)
    z_bins = np.linspace(points[:, 2].min(), points[:, 2].max(), n_z + 1)
    hist, _, _ = np.histogram2d(
        np.arctan2(points[:, 1], points[:, 0]), points[:, 2], bins=(theta_bins, z_bins)
    )
    return np.std(hist) / np.mean(hist)


def main() -> None:
    sampler = FastCylindricalSampler(length=100.0, radius=10.0)
    pts = sampler.sample(10000)
    cv = coverage_cv(pts)
    print(f"Coverage CV: {cv:.3f}")
    assert cv < 0.15, "Surface coverage is non-uniform"


if __name__ == "__main__":
    main()
