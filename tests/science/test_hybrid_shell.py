import numpy as np

from fluxmd.core.solvent.hybrid_shell import HybridSolventShell, water_count_in_shell


def test_water_count_in_shell():
    """Water count matches expected for simple coordinates."""
    center = np.array([0.0, 0.0, 0.0])
    waters = np.array(
        [
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.9, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        ]
    )

    count = water_count_in_shell(waters, center, inner_radius=2.0, outer_radius=5.0)
    assert count == 2

    shell = HybridSolventShell(waters, center, 2.0, 5.0)
    assert shell.count_waters() == 2
