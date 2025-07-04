"""
Edge case tests for Matryoshka trajectory engine.

Tests handling of unusual inputs, boundary conditions, and error scenarios.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest

from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.surface.layer_stream import MatryoshkaLayerGenerator
from fluxmd.core.surface.ses_builder import SESBuilder, SurfaceMesh


class TestDegenerateInputs:
    """Test handling of degenerate or unusual inputs."""

    def test_single_atom_protein(self):
        """Test with a protein containing only one atom."""
        protein_atoms = {
            "coords": np.array([[0, 0, 0]]),
            "names": np.array(["CA"]),
            "radii": np.array([1.8]),
            "masses": np.array([12.0]),
            "resnames": np.array(["ALA"]),
        }

        ligand_atoms = {
            "coords": np.array([[5, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False, "n_workers": 1}

        # Should handle gracefully
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        assert generator is not None

    def test_colinear_atoms(self):
        """Test with all atoms in a straight line."""
        protein_atoms = {
            "coords": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]),
            "names": np.array(["CA", "CA", "CA", "CA"]),
            "radii": np.ones(4) * 1.8,
            "masses": np.ones(4) * 12.0,
            "resnames": np.array(["ALA"] * 4),
        }

        ligand_atoms = {
            "coords": np.array([[5, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False, "n_workers": 1}

        # PCA anchors should still work
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        anchor_dist = np.linalg.norm(generator.anchors[1] - generator.anchors[0])
        assert anchor_dist > 0, "Anchors collapsed to same point"

    def test_empty_ligand(self):
        """Test with empty ligand atoms."""
        protein_atoms = {
            "coords": np.random.rand(10, 3) * 10,
            "names": np.array(["CA"] * 10),
            "radii": np.ones(10) * 1.8,
            "masses": np.ones(10) * 12.0,
            "resnames": np.array(["ALA"] * 10),
        }

        ligand_atoms = {
            "coords": np.array([]).reshape(0, 3),
            "names": np.array([]),
            "masses": np.array([]),
        }

        params = {"use_ref15": False, "n_workers": 1}

        # Should raise an error
        with pytest.raises(Exception):
            generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

    def test_nan_coordinates(self):
        """Test handling of NaN coordinates."""
        protein_atoms = {
            "coords": np.array([[0, 0, 0], [np.nan, 0, 0], [0, 5, 0]]),
            "names": np.array(["CA", "CA", "CA"]),
            "radii": np.ones(3) * 1.8,
            "masses": np.ones(3) * 12.0,
            "resnames": np.array(["ALA"] * 3),
        }

        ligand_atoms = {
            "coords": np.array([[5, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False, "n_workers": 1}

        # Should handle or raise meaningful error
        with pytest.raises(Exception):
            generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)


class TestSurfaceEdgeCases:
    """Test edge cases in surface generation."""

    def test_self_intersecting_surface(self):
        """Test handling of self-intersecting surfaces."""
        # Create a surface that will self-intersect when offset
        vertices = np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [1, 0.1, 0],  # Nearly colinear triangle
                [1, 0, 1],
                [1, 0, -1],  # Points above and below
            ]
        )
        faces = np.array([[0, 1, 2], [0, 2, 3], [1, 2, 4]])

        base_surface = SurfaceMesh(vertices, faces)
        layer_gen = MatryoshkaLayerGenerator(base_surface, step=5.0)  # Large step

        # Get offset layer - should handle self-intersection
        layer_2 = layer_gen.get_layer(2)
        assert layer_2 is not None
        assert len(layer_2.vertices) == len(vertices)

    def test_surface_with_holes(self):
        """Test surface generation with non-manifold geometry."""
        # Create a surface with a hole (missing faces)
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]]  # Square  # Peak
        )
        faces = np.array(
            [
                [0, 1, 4],
                [1, 2, 4],  # Two faces only, creating a hole
            ]
        )

        surface = SurfaceMesh(vertices, faces)
        builder = SESBuilder(vertices[:4], np.ones(4) * 1.8)  # Use first 4 vertices

        # Should handle incomplete surface
        assert surface.vertices.shape[0] == 5
        assert surface.faces.shape[0] == 2

    def test_extreme_layer_offsets(self):
        """Test very large layer offsets."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        surface = SurfaceMesh(vertices, faces)

        layer_gen = MatryoshkaLayerGenerator(surface, step=0.1)

        # Test very large layer index
        large_layer = layer_gen.get_layer(1000)

        # Vertices should be offset by 100 Angstroms
        expected_offset = 100.0
        actual_offsets = np.linalg.norm(large_layer.vertices - vertices, axis=1)

        # Check offset is approximately correct (within 10%)
        assert all(
            abs(offset - expected_offset) / expected_offset < 0.1 for offset in actual_offsets
        )


class TestParallelProcessingEdgeCases:
    """Test edge cases in parallel processing."""

    def test_worker_crash_recovery(self):
        """Test recovery from worker process crashes."""
        protein_atoms = {
            "coords": np.random.rand(10, 3) * 10,
            "names": np.array(["CA"] * 10),
            "radii": np.ones(10) * 1.8,
            "masses": np.ones(10) * 12.0,
            "resnames": np.array(["ALA"] * 10),
        }

        ligand_atoms = {
            "coords": np.random.rand(5, 3),
            "names": np.array(["C"] * 5),
            "masses": np.ones(5) * 12.0,
        }

        params = {"use_ref15": False, "n_workers": 2, "max_steps": 10}  # Multiple workers

        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

        # Mock a worker crash
        original_run = generator._run_single_trajectory
        call_count = 0

        def mock_run(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Crash on second call
                raise RuntimeError("Simulated worker crash")
            return original_run(*args, **kwargs)

        with patch.object(generator, "_run_single_trajectory", side_effect=mock_run):
            # Should handle the error gracefully
            trajectories = generator.run(n_layers=1, n_iterations=3)

            # Should have results despite one crash
            assert len(trajectories) >= 2  # At least some succeeded

    def test_zero_workers(self):
        """Test with n_workers=0 (should use serial processing)."""
        protein_atoms = {
            "coords": np.random.rand(10, 3) * 10,
            "names": np.array(["CA"] * 10),
            "radii": np.ones(10) * 1.8,
            "masses": np.ones(10) * 12.0,
            "resnames": np.array(["ALA"] * 10),
        }

        ligand_atoms = {
            "coords": np.random.rand(5, 3),
            "names": np.array(["C"] * 5),
            "masses": np.ones(5) * 12.0,
        }

        params = {"use_ref15": False, "n_workers": 0, "max_steps": 10}  # Force serial

        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        trajectories = generator.run(n_layers=1, n_iterations=2)

        assert len(trajectories) == 2  # Should complete normally


class TestCheckpointingEdgeCases:
    """Test edge cases in checkpointing."""

    def test_corrupt_checkpoint(self):
        """Test recovery from corrupted checkpoint files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a corrupt checkpoint file
            checkpoint_path = os.path.join(tmpdir, "checkpoint_L0_I0.pkl")
            with open(checkpoint_path, "wb") as f:
                f.write(b"corrupted data")

            protein_atoms = {
                "coords": np.random.rand(10, 3) * 10,
                "names": np.array(["CA"] * 10),
                "radii": np.ones(10) * 1.8,
                "masses": np.ones(10) * 12.0,
                "resnames": np.array(["ALA"] * 10),
            }

            ligand_atoms = {
                "coords": np.random.rand(5, 3),
                "names": np.array(["C"] * 5),
                "masses": np.ones(5) * 12.0,
            }

            params = {"use_ref15": False, "n_workers": 1, "checkpoint_dir": tmpdir, "max_steps": 10}

            # Should handle corrupt checkpoint gracefully
            generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
            trajectories = generator.run(n_layers=1, n_iterations=1)

            assert len(trajectories) > 0  # Should complete despite corruption

    def test_checkpoint_with_full_disk(self):
        """Test behavior when checkpoint directory is full."""
        protein_atoms = {
            "coords": np.random.rand(10, 3) * 10,
            "names": np.array(["CA"] * 10),
            "radii": np.ones(10) * 1.8,
            "masses": np.ones(10) * 12.0,
            "resnames": np.array(["ALA"] * 10),
        }

        ligand_atoms = {
            "coords": np.random.rand(5, 3),
            "names": np.array(["C"] * 5),
            "masses": np.ones(5) * 12.0,
        }

        # Use a non-existent directory to simulate write failure
        params = {
            "use_ref15": False,
            "n_workers": 1,
            "checkpoint_dir": "/nonexistent/directory",
            "max_steps": 10,
        }

        # Should continue without checkpointing
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        trajectories = generator.run(n_layers=1, n_iterations=1)

        assert len(trajectories) > 0  # Should complete despite checkpoint failure


class TestEnergyCalculationEdgeCases:
    """Test edge cases in energy calculations."""

    def test_overlapping_atoms(self):
        """Test energy calculation with overlapping atoms."""
        from fluxmd.core.ref15_energy import AtomContext, REF15EnergyCalculator

        calculator = REF15EnergyCalculator()

        # Create overlapping atoms
        atom1 = AtomContext(atom_type="C", coords=np.array([0, 0, 0]), formal_charge=0.0)

        atom2 = AtomContext(
            atom_type="C", coords=np.array([0.001, 0, 0]), formal_charge=0.0  # Nearly overlapping
        )

        # Should handle very small distances
        energy = calculator.calculate_interaction_energy(atom1, atom2, 0.001)
        assert np.isfinite(energy), f"Non-finite energy for overlapping atoms: {energy}"

    def test_extreme_temperatures(self):
        """Test Brownian dynamics at extreme temperatures."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        surface = SurfaceMesh(vertices, faces)

        ligand_sphere = {
            "radius": 1.0,
            "mass": 12.0,
            "center": np.array([0.5, 0.5, 1.0]),
            "inertia": 2.0,
        }

        # Test at very low temperature
        from fluxmd.core.dynamics.brownian_roller import BrownianSurfaceRoller

        roller_cold = BrownianSurfaceRoller(
            surface=surface,
            ligand_sphere=ligand_sphere,
            anchors=(np.array([0, 0, 0]), np.array([1, 1, 0])),
            T=1.0,  # 1 Kelvin
            seed=42,
        )

        traj_cold = roller_cold.run(max_steps=100)

        # Test at very high temperature
        roller_hot = BrownianSurfaceRoller(
            surface=surface,
            ligand_sphere=ligand_sphere,
            anchors=(np.array([0, 0, 0]), np.array([1, 1, 0])),
            T=1000.0,  # 1000 Kelvin
            seed=42,
        )

        traj_hot = roller_hot.run(max_steps=100)

        # Cold trajectory should have less motion
        pos_cold = np.array(traj_cold["pos"])
        pos_hot = np.array(traj_hot["pos"])

        var_cold = np.var(pos_cold, axis=0).sum()
        var_hot = np.var(pos_hot, axis=0).sum()

        assert var_hot > var_cold, "High temperature should have more motion"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
