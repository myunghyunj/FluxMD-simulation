import os
import sys
import hashlib
import numpy as np

np.random.seed(0)
import pandas as pd
import torch

# Allow running tests without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fluxmd.core.trajectory_generator import ProteinLigandFluxAnalyzer

SEED = 1337
EXPECTED_HASH = "40376662bb4c2077183a7bf006a98823c165b74f40be2a1eb496a19c92cafd51"


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
    return torch.tensor(np.linalg.norm(traj, axis=1), dtype=torch.float64, device=device)


def test_flux_hash_stable(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    flux = simple_trajectory_generator(device)
    assert _hash(flux) == EXPECTED_HASH
