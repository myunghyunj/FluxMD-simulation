import numpy as np

from fluxmd.core.cylindrical_sampler import FastCylindricalSampler

np.random.seed(0)


def test_sampling_cv_large():
    pts = FastCylindricalSampler(120.0, 12.0).sample(200000)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    z = pts[:, 2]
    hist, _ = np.histogramdd((theta, z), bins=(24, 24))
    cv = np.std(hist) / np.mean(hist)
    assert cv < 0.15
