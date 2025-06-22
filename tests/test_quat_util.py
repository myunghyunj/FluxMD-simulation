import numpy as np
from fluxmd.core.dynamics.brownian_roller import quaternion_multiply

def test_quaternion_identity():
    q = np.array([1.0,0.0,0.0,0.0])
    r = quaternion_multiply(q, q)
    assert np.allclose(r, q) 