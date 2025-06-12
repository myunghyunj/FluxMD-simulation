import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import hashlib
import json
import numpy as np
import torch
import pandas as pd
import pytest

from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer

SEED = 13_37


def _hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def simple_trajectory_generator(device: str = "cpu") -> torch.Tensor:
    analyzer = ProteinLigandFluxAnalyzer(
        protein_file="dummy.pdb",
        ligand_file="dummy.pdb",
        output_dir=".",
    )
    analyzer.collision_detector.protein_radii = np.array([1.7])
    analyzer.collision_detector.protein_tree = type(
        "dummy",
        (),
        {"query_ball_point": lambda *args, **kwargs: []},
    )()
    ligand_coords = np.array([[0.0, 0.0, 0.0]])
    ligand_atoms = pd.DataFrame({"name": ["C"]})
    traj = analyzer.generate_random_walk_trajectory(
        start_pos=np.zeros(3),
        n_steps=10,
        ligand_coords=ligand_coords,
        ligand_atoms=ligand_atoms,
        molecular_weight=300.0,
        dt=40,
        max_distance=5.0,
        max_attempts=1,
    )
    return torch.tensor(np.linalg.norm(traj, axis=1))


def test_flux_hash_stable(device, tmp_path):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    flux = simple_trajectory_generator(device)

    golden_file = tmp_path / "golden_hash.json"
    h = _hash(flux)
    if golden_file.exists() and not bool(
        getattr(pytest, "config", None) and pytest.config.getoption("--update-golden")
    ):
        assert json.load(golden_file.open())["hash"] == h
    else:
        json.dump({"hash": h}, golden_file.open("w"))
