"""
fluxmd.core.cylindrical_sampler
--------------------------------
Stateless generator yielding points uniformly distributed on the surface of a finite cylinder. Includes optional groove bias and CuPy fall-back for GPU arrays.
"""

from __future__ import annotations

import numpy as np
import warnings

try:  # GPU-friendly array module
    import cupy as xp
    CUPY_OK = True
except Exception:  # pragma: no cover - CuPy not installed
    import numpy as xp  # type: ignore
    CUPY_OK = False


class FastCylindricalSampler:
    def __init__(self, length: float, radius: float, pad: float = 0.1,
                 groove_bias: float | None = None) -> None:
        if length <= 0 or radius <= 0:
            raise ValueError("length and radius must be positive")
        self._L = float(length)
        self._R = float(radius)
        self._z0 = -pad * length
        self._z1 = (1 + pad) * length
        self._alpha = groove_bias

    def _sample_loop(
        self, n: int, groove_angles: tuple[float, float] | None = None
    ) -> np.ndarray:
        """Legacy loop sampler (deprecated)."""
        warnings.warn(
            "_sample_loop is deprecated; use vectorised sampling",
            DeprecationWarning,
            stacklevel=2,
        )
        rng = np.random.default_rng()
        pts = xp.empty((n, 3), dtype=xp.float32)
        acc = 0
        while acc < n:
            theta = rng.uniform(0, 2 * xp.pi)
            if self._alpha is not None and groove_angles is not None:
                w = 1.0 + self._alpha * (
                    xp.cos(theta - groove_angles[0])
                    + xp.cos(theta - groove_angles[1])
                )
                if rng.random() >= w / (1 + 2 * self._alpha):
                    continue
            r = self._R * xp.sqrt(rng.random())
            z = rng.uniform(self._z0, self._z1)
            pts[acc] = (r * xp.cos(theta), r * xp.sin(theta), z)
            acc += 1
        return xp.asarray(pts)

    def _sample_vectorised(
        self, n: int, groove_angles: tuple[float, float] | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng()
        batch = n * 2
        theta = rng.uniform(0, 2 * xp.pi, batch)
        r = self._R * xp.sqrt(rng.random(batch))
        z = rng.uniform(self._z0, self._z1, batch)
        if self._alpha is not None and groove_angles is not None:
            w = 1.0 + self._alpha * (
                xp.cos(theta - groove_angles[0])
                + xp.cos(theta - groove_angles[1])
            )
            keep = rng.random(batch) < w / (1 + 2 * self._alpha)
            theta, r, z = theta[keep], r[keep], z[keep]
        pts = xp.column_stack(
            (r[:n] * xp.cos(theta[:n]), r[:n] * xp.sin(theta[:n]), z[:n])
        ).astype(xp.float32)
        if not CUPY_OK:
            return xp.asarray(pts)
        return pts

    def sample(
        self,
        n: int,
        groove_angles: tuple[float, float] | None = None,
        *,
        vectorised: bool = True,
    ) -> np.ndarray:
        """Return *n* points in Cartesian coordinates."""
        if vectorised:
            return self._sample_vectorised(n, groove_angles)
        else:
            return self._sample_loop(n, groove_angles)

