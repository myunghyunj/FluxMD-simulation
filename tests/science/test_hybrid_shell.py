import numpy as np
from fluxmd.core.solvent.hybrid_shell import build_hybrid_shell


def test_scaffold_runs():
    lig = np.zeros((10, 3))
    rec = np.zeros((100, 3))
    waters = build_hybrid_shell(lig, rec)
    assert waters.shape[1] == 3
