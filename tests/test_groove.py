import math
import numpy as np

np.random.seed(0)

from fluxmd.utils.dna_to_pdb import groove_vectors


def test_groove_angle():
    minor, major = groove_vectors(0.0)
    deg = math.degrees(np.arccos(minor @ major))
    assert 155 < deg < 165

