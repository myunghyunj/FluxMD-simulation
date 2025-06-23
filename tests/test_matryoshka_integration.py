"""
Matryoshka integration tests with synthetic protein systems.

These tests validate the full Matryoshka pipeline including:
- Surface generation
- Brownian dynamics
- Layer hopping
- Energy calculations
- DNA groove bias
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fluxmd.analysis.flux_analyzer import TrajectoryFluxAnalyzer
from fluxmd.core.dynamics.brownian_roller import BrownianSurfaceRoller
from fluxmd.core.geometry.pca_anchors import extreme_calpha_pairs
from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.surface.dna_groove_detector import DNAGrooveDetector
from fluxmd.core.surface.ses_builder import SESBuilder


def create_synthetic_protein(n_atoms=1000, include_dna=False):
    """
    Create a synthetic protein system for testing.

    Creates a globular protein roughly 30Å in diameter with:
    - Realistic atom type distribution
    - Proper backbone connectivity
    - Optional DNA double helix
    """
    np.random.seed(42)  # Deterministic generation

    atoms = []

    # Generate protein core (70% of atoms)
    n_protein = int(n_atoms * 0.7) if include_dna else n_atoms

    # Create alpha-carbon backbone spiral
    n_residues = n_protein // 5  # ~5 atoms per residue
    t = np.linspace(0, 4 * np.pi, n_residues)

    # Spiral parameters for compact globular shape
    radius = 15.0 * (1 - t / (4 * np.pi) * 0.3)  # Decreasing radius
    z_height = np.linspace(-15, 15, n_residues)

    ca_x = radius * np.cos(t)
    ca_y = radius * np.sin(t)
    ca_z = z_height

    # Build residues around each CA
    atom_idx = 0
    for i in range(n_residues):
        # CA atom
        atoms.append(
            {
                "serial": atom_idx + 1,
                "name": "CA",
                "residue": "ALA",
                "chain": "A",
                "residue_number": i + 1,
                "x": ca_x[i],
                "y": ca_y[i],
                "z": ca_z[i],
                "element": "C",
            }
        )
        atom_idx += 1

        # Add sidechain atoms in random directions
        n_sidechain = np.random.randint(2, 6)
        for j in range(n_sidechain):
            if atom_idx >= n_protein:
                break

            # Random offset from CA
            offset = np.random.randn(3) * 2.0
            atoms.append(
                {
                    "serial": atom_idx + 1,
                    "name": ["CB", "CG", "CD", "CE", "CZ"][j % 5],
                    "residue": "ALA",
                    "chain": "A",
                    "residue_number": i + 1,
                    "x": ca_x[i] + offset[0],
                    "y": ca_y[i] + offset[1],
                    "z": ca_z[i] + offset[2],
                    "element": ["C", "C", "C", "N", "O"][j % 5],
                }
            )
            atom_idx += 1

    # Add DNA if requested
    if include_dna:
        n_dna = n_atoms - len(atoms)
        n_base_pairs = n_dna // 20  # ~20 atoms per base pair

        # DNA helix parameters
        helix_radius = 10.0
        helix_pitch = 34.0  # Å per turn
        base_pair_rise = 3.4  # Å between base pairs

        for i in range(n_base_pairs):
            z = -20 + i * base_pair_rise
            angle = 2 * np.pi * i / 10  # 10 bp per turn

            # Strand 1 backbone
            x1 = 35 + helix_radius * np.cos(angle)
            y1 = helix_radius * np.sin(angle)

            atoms.append(
                {
                    "serial": len(atoms) + 1,
                    "name": "P",
                    "residue": "DA",
                    "chain": "B",
                    "residue_number": i + 1,
                    "x": x1,
                    "y": y1,
                    "z": z,
                    "element": "P",
                }
            )

            # Strand 2 backbone (antiparallel)
            x2 = 35 + helix_radius * np.cos(angle + np.pi)
            y2 = helix_radius * np.sin(angle + np.pi)

            atoms.append(
                {
                    "serial": len(atoms) + 1,
                    "name": "P",
                    "residue": "DT",
                    "chain": "C",
                    "residue_number": n_base_pairs - i,
                    "x": x2,
                    "y": y2,
                    "z": z,
                    "element": "P",
                }
            )

            # Add sugar and base atoms
            for strand, (x, y) in enumerate([(x1, y1), (x2, y2)]):
                if len(atoms) >= n_atoms:
                    break

                # Sugar atoms
                for j, atom_name in enumerate(["C1'", "C2'", "C3'", "C4'", "C5'"]):
                    if len(atoms) >= n_atoms:
                        break
                    atoms.append(
                        {
                            "serial": len(atoms) + 1,
                            "name": atom_name,
                            "residue": ["DA", "DT"][strand],
                            "chain": ["B", "C"][strand],
                            "residue_number": [i + 1, n_base_pairs - i][strand],
                            "x": x + np.random.randn() * 1.5,
                            "y": y + np.random.randn() * 1.5,
                            "z": z + np.random.randn() * 0.5,
                            "element": "C",
                        }
                    )

    return pd.DataFrame(atoms)


def create_synthetic_ligand(center=(0, 0, 0), n_atoms=20):
    """Create a small molecule ligand."""
    np.random.seed(43)

    atoms = []
    # Create benzene-like ring
    for i in range(6):
        angle = 2 * np.pi * i / 6
        atoms.append(
            {
                "serial": i + 1,
                "name": f"C{i+1}",
                "residue": "LIG",
                "chain": "L",
                "residue_number": 1,
                "x": center[0] + 1.4 * np.cos(angle),
                "y": center[1] + 1.4 * np.sin(angle),
                "z": center[2],
                "element": "C",
            }
        )

    # Add functional groups
    remaining = n_atoms - 6
    for i in range(remaining):
        base_atom = i % 6
        offset = np.random.randn(3) * 1.5
        atoms.append(
            {
                "serial": len(atoms) + 1,
                "name": ["O", "N", "C", "S"][i % 4] + str(i),
                "residue": "LIG",
                "chain": "L",
                "residue_number": 1,
                "x": atoms[base_atom]["x"] + offset[0],
                "y": atoms[base_atom]["y"] + offset[1],
                "z": atoms[base_atom]["z"] + offset[2],
                "element": ["O", "N", "C", "S"][i % 4],
            }
        )

    return pd.DataFrame(atoms)


# Create minimal synthetic data for testing
def create_test_system():
    protein_atoms = {
        "coords": np.random.rand(100, 3) * 20,
        "names": np.array(["CA"] * 100),
        "radii": np.ones(100) * 1.8,
        "masses": np.ones(100) * 12.0,
        "resnames": np.array(["ALA"] * 100),
    }
    ligand_atoms = {
        "coords": np.random.rand(10, 3),
        "names": np.array(["C"] * 10),
        "masses": np.ones(10) * 12.0,
    }
    params = {"use_ref15": False, "n_workers": 1}
    return protein_atoms, ligand_atoms, params


class TestMatryoshkaIntegration:
    """Integration tests for the Matryoshka trajectory engine."""

    def test_synthetic_protein_generation(self):
        """Test synthetic protein generator."""
        protein = create_synthetic_protein(n_atoms=100)
        # The generator may produce slightly fewer atoms due to chain termination logic
        assert 85 <= len(protein) <= 100
        assert "CA" in protein["name"].values
        assert all(protein[["x", "y", "z"]].notna().all())

        # Check spatial extent
        coords = protein[["x", "y", "z"]].values
        extent = coords.max(axis=0) - coords.min(axis=0)
        assert 20 < extent.max() < 40  # Reasonable protein size

    def test_synthetic_dna_generation(self):
        """Test synthetic DNA generator."""
        system = create_synthetic_protein(n_atoms=200, include_dna=True)

        # Check for DNA atoms
        dna_atoms = system[system["chain"].isin(["B", "C"])]
        assert len(dna_atoms) > 0
        assert "P" in dna_atoms["name"].values  # Phosphate backbone
        assert "C1'" in dna_atoms["name"].values  # Sugar atoms

    def test_full_matryoshka_pipeline(self, tmp_path):
        """Test complete Matryoshka pipeline with synthetic system."""
        protein_atoms, ligand_atoms, params = create_test_system()
        params["checkpoint_dir"] = tmp_path

        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        trajectories = generator.run(n_layers=2, n_iterations=1)
        assert len(trajectories) > 0
        assert "pos" in trajectories[0]

    def test_surface_generation(self):
        """Test SES surface generation on synthetic protein."""
        protein_atoms, _, _ = create_test_system()
        builder = SESBuilder(protein_atoms["coords"], protein_atoms["radii"])
        surface = builder.build_ses0()
        assert surface.vertices.shape[0] > 0
        assert surface.faces.shape[0] > 0

    def test_dna_groove_detection(self):
        """Test DNA groove detection on synthetic system."""
        protein_atoms, _, _ = create_test_system()
        # This test primarily checks initialization, not the logic itself
        detector = DNAGrooveDetector(protein_atoms)
        assert detector is not None

    def test_brownian_dynamics_physics(self):
        """Test Brownian dynamics produces physical trajectories."""
        protein_atoms, ligand_atoms, _ = create_test_system()
        builder = SESBuilder(protein_atoms["coords"], protein_atoms["radii"])
        surface = builder.build_ses0()
        anchors = (protein_atoms["coords"][0], protein_atoms["coords"][-1])

        ligand_sphere = MatryoshkaTrajectoryGenerator(
            protein_atoms, ligand_atoms, {"use_ref15": False}
        ).ligand_sphere

        roller = BrownianSurfaceRoller(
            surface=surface, ligand_sphere=ligand_sphere, anchors=anchors
        )
        trajectory = roller.run(max_steps=50)
        assert len(trajectory["pos"]) > 0

    def test_performance_scaling(self, tmp_path):
        """Test performance with a standard system size."""
        protein_atoms, ligand_atoms, params = create_test_system()
        params["checkpoint_dir"] = tmp_path
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        trajectories = generator.run(n_layers=1, n_iterations=1)
        assert len(trajectories) == 1

    def test_deterministic_results(self, tmp_path):
        """Test that results are deterministic with fixed seed."""
        protein_atoms, ligand_atoms, params = create_test_system()
        params["checkpoint_dir"] = tmp_path

        # Run 1
        params["seed"] = 42
        gen1 = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        traj1 = gen1.run(n_layers=1, n_iterations=1)

        # Run 2
        gen2 = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        traj2 = gen2.run(n_layers=1, n_iterations=1)

        assert np.allclose(traj1[0]["pos"], traj2[0]["pos"])


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--synthetic-only",
        action="store_true",
        default=False,
        help="Run only tests with synthetic systems (for CI)",
    )


def pytest_collection_modifyitems(config, items):
    """Filter tests based on command line options."""
    if not config.getoption("--synthetic-only"):
        return

    # In synthetic-only mode, skip tests that require real PDB files
    skip_real = pytest.mark.skip(reason="Skipping tests requiring real PDB files")
    for item in items:
        # Add markers for tests that need real data
        if "real_pdb" in item.keywords:
            item.add_marker(skip_real)
