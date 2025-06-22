import numpy as np
import pytest
from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator


def _dummy_structures():
    protein_atoms = {
        "coords": np.random.rand(5, 3),
        "names": ["CA"] * 5,
        "radii": np.ones(5) * 1.8,
        "masses": np.ones(5) * 12.0,
        "resnames": ["ALA"] * 5,
    }
    ligand_atoms = {
        "coords": np.random.rand(3, 3),
        "names": ["C"] * 3,
        "masses": np.ones(3) * 12.0,
    }
    return protein_atoms, ligand_atoms


@pytest.mark.parametrize("value", [None, "", "auto"])
def test_auto_values(value):
    protein, ligand = _dummy_structures()
    gen = MatryoshkaTrajectoryGenerator(protein, ligand, {"n_workers": value, "use_ref15": False})
    assert isinstance(gen.n_workers, int) and gen.n_workers >= 1


def test_invalid_string_raises():
    protein, ligand = _dummy_structures()
    with pytest.raises(ValueError):
        MatryoshkaTrajectoryGenerator(protein, ligand, {"n_workers": "invalid", "use_ref15": False})


@pytest.mark.parametrize("value", [0, -5])
def test_non_positive_clamped(value):
    protein, ligand = _dummy_structures()
    gen = MatryoshkaTrajectoryGenerator(protein, ligand, {"n_workers": value, "use_ref15": False})
    assert gen.n_workers == 1


def test_valid_integer():
    protein, ligand = _dummy_structures()
    gen = MatryoshkaTrajectoryGenerator(protein, ligand, {"n_workers": 4, "use_ref15": False})
    assert gen.n_workers == 4
