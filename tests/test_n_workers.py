import os
import numpy as np
import pytest

from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.surface import SurfaceMesh

class DummySESBuilder:
    def __init__(self, *args, **kwargs):
        pass
    def build_ses0(self):
        return SurfaceMesh(np.zeros((3,3)), np.array([[0,1,2]], dtype=np.int32))

class DummyLayerGenerator:
    def __init__(self, base_surface, step):
        self.base_surface = base_surface
        self.step = step
    def get_max_useful_layers(self, ligand_radius, cutoff=12.0):
        return 1

@pytest.fixture(autouse=True)
def patch_generator(monkeypatch):
    monkeypatch.setattr('fluxmd.core.matryoshka_generator.SESBuilder', DummySESBuilder)
    monkeypatch.setattr('fluxmd.core.matryoshka_generator.MatryoshkaLayerGenerator', DummyLayerGenerator)
    monkeypatch.setattr('fluxmd.core.matryoshka_generator.extreme_calpha_pairs',
                        lambda coords, names: (np.zeros(3), np.ones(3)))
    monkeypatch.setattr('fluxmd.core.matryoshka_generator.get_ref15_calculator', lambda pH: None)
    yield


def minimal_atoms():
    protein_atoms = {
        'coords': np.array([[0.0, 0.0, 0.0]]),
        'names': np.array(['CA']),
        'radii': np.array([1.5]),
        'masses': np.array([12.0]),
        'resnames': np.array(['ALA'])
    }
    ligand_atoms = {
        'coords': np.array([[0.0, 0.0, 0.0]]),
        'names': np.array(['C']),
        'masses': np.array([12.0])
    }
    return protein_atoms, ligand_atoms


@pytest.mark.parametrize('input_value, expected', [
    (None, 7),
    ('', 7),
    ('auto', 7),
    (0, 1),
    (-5, 1),
    (4, 4)
])
def test_n_workers_parsing(monkeypatch, input_value, expected):
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    protein_atoms, ligand_atoms = minimal_atoms()
    gen = MatryoshkaTrajectoryGenerator(
        protein_atoms,
        ligand_atoms,
        {'n_workers': input_value, 'use_ref15': False}
    )
    assert gen.n_workers == expected


def test_n_workers_invalid(monkeypatch):
    monkeypatch.setattr(os, 'cpu_count', lambda: 8)
    protein_atoms, ligand_atoms = minimal_atoms()
    with pytest.raises(ValueError):
        MatryoshkaTrajectoryGenerator(
            protein_atoms,
            ligand_atoms,
            {'n_workers': 'invalid_string', 'use_ref15': False}
        )
