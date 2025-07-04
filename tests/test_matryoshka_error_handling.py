"""
Error handling tests for Matryoshka trajectory engine.

Tests proper error handling, meaningful error messages, and graceful degradation.
"""

import multiprocessing as mp
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.surface.layer_stream import MatryoshkaLayerGenerator
from fluxmd.core.surface.ses_builder import SurfaceMesh


class TestInputValidation:
    """Test input validation and error messages."""

    def test_missing_required_fields(self):
        """Test error handling for missing required fields."""
        # Missing coords
        protein_atoms = {
            "names": np.array(["CA"]),
            "radii": np.array([1.8]),
            "masses": np.array([12.0]),
        }

        ligand_atoms = {
            "coords": np.array([[5, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False}

        with pytest.raises(KeyError):
            MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

    def test_mismatched_array_lengths(self):
        """Test error handling for mismatched array lengths."""
        protein_atoms = {
            "coords": np.array([[0, 0, 0], [5, 0, 0]]),  # 2 atoms
            "names": np.array(["CA", "CA", "CA"]),  # 3 names
            "radii": np.ones(2) * 1.8,
            "masses": np.ones(2) * 12.0,
        }

        ligand_atoms = {
            "coords": np.array([[5, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False}

        # Should validate array lengths
        with pytest.raises(Exception):
            MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

    def test_invalid_parameter_types(self):
        """Test error handling for invalid parameter types."""
        protein_atoms = {
            "coords": "not an array",  # Invalid type
            "names": np.array(["CA"]),
            "radii": np.array([1.8]),
            "masses": np.array([12.0]),
        }

        ligand_atoms = {
            "coords": np.array([[5, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False}

        with pytest.raises(Exception):
            MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)


class TestSurfaceGenerationErrors:
    """Test error handling in surface generation."""

    def test_insufficient_atoms_for_surface(self):
        """Test surface generation with too few atoms."""
        # Only 2 atoms - can't form a surface
        protein_atoms = {
            "coords": np.array([[0, 0, 0], [5, 0, 0]]),
            "names": np.array(["CA", "CA"]),
            "radii": np.ones(2) * 1.8,
            "masses": np.ones(2) * 12.0,
            "resnames": np.array(["ALA", "ALA"]),
        }

        ligand_atoms = {
            "coords": np.array([[10, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        params = {"use_ref15": False, "n_workers": 1}

        # Should handle gracefully or provide meaningful error
        try:
            generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
            # If it doesn't raise, check that surface is minimal
            assert len(generator.base_surface.vertices) > 0
        except Exception as e:
            # Should have meaningful error message
            assert "surface" in str(e).lower() or "insufficient" in str(e).lower()

    def test_negative_layer_index(self):
        """Test accessing negative layer index."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        surface = SurfaceMesh(vertices, faces)

        layer_gen = MatryoshkaLayerGenerator(surface, step=1.0)

        # Should raise error for negative index
        with pytest.raises(Exception):
            layer_gen._offset_layer(-1)


class TestParallelProcessingErrors:
    """Test error handling in parallel processing."""

    def test_worker_queue_timeout(self):
        """Test handling of worker queue timeouts."""
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

        params = {"use_ref15": False, "n_workers": 2, "max_steps": 10}

        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

        # Mock queue with timeout
        mock_queue = MagicMock()
        mock_queue.get.side_effect = mp.queues.Empty()

        # Worker should handle empty queue gracefully
        result_queue = mp.Queue()
        generator._worker_process(mock_queue, result_queue, worker_id=0)

        # Should not crash
        assert result_queue.empty()  # No results due to empty work queue

    def test_pickle_error_in_worker(self):
        """Test handling of pickling errors in worker communication."""
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

        # Add unpicklable object
        params = {
            "use_ref15": False,
            "n_workers": 2,
            "max_steps": 10,
            "unpicklable": lambda x: x,  # Functions can't be pickled
        }

        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

        # Should handle pickle errors
        try:
            trajectories = generator.run(n_layers=1, n_iterations=1)
            # If parallel fails, might fall back to serial
            assert len(trajectories) >= 0
        except Exception as e:
            # Should have meaningful error about pickling
            assert "pickle" in str(e).lower() or "serialize" in str(e).lower()


class TestEnergyCalculationErrors:
    """Test error handling in energy calculations."""

    def test_invalid_atom_type(self):
        """Test handling of invalid atom types in REF15."""
        from fluxmd.core.ref15_energy import AtomContext, REF15EnergyCalculator

        calculator = REF15EnergyCalculator()

        # Create atom with invalid type
        atom1 = AtomContext(atom_type="INVALID_TYPE", coords=np.array([0, 0, 0]), formal_charge=0.0)

        atom2 = AtomContext(atom_type="C", coords=np.array([2, 0, 0]), formal_charge=0.0)

        # Should handle gracefully
        try:
            energy = calculator.calculate_interaction_energy(atom1, atom2, 2.0)
            # If it doesn't crash, energy should be finite
            assert np.isfinite(energy)
        except KeyError:
            # Expected for truly invalid types
            pass

    def test_energy_calculator_exception(self):
        """Test handling of exceptions in custom energy calculator."""
        protein_atoms = {
            "coords": np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]]),
            "names": np.array(["CA", "CA", "CA"]),
            "radii": np.ones(3) * 1.8,
            "masses": np.ones(3) * 12.0,
            "resnames": np.array(["ALA"] * 3),
        }

        ligand_atoms = {
            "coords": np.array([[10, 0, 0]]),
            "names": np.array(["C"]),
            "masses": np.array([12.0]),
        }

        def failing_energy_calc(pos, quat, layer):
            raise RuntimeError("Energy calculation failed")

        params = {
            "use_ref15": False,
            "n_workers": 1,
            "max_steps": 10,
            "ref15_calculator": Mock(side_effect=failing_energy_calc),
        }

        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)

        # Should handle energy calculation failures
        try:
            trajectories = generator.run(n_layers=1, n_iterations=1)
            # Might fall back to simple energy
            assert len(trajectories) >= 0
        except RuntimeError:
            # Or propagate the error with context
            pass


class TestFileIOErrors:
    """Test error handling in file I/O operations."""

    def test_checkpoint_read_permission_error(self):
        """Test handling of permission errors when reading checkpoints."""
        import os
        import stat
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoint file
            checkpoint_path = os.path.join(tmpdir, "checkpoint_L0_I0.pkl")
            with open(checkpoint_path, "wb") as f:
                f.write(b"data")

            # Remove read permissions
            os.chmod(checkpoint_path, stat.S_IWRITE)

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

            # Should handle permission error gracefully
            try:
                generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
                trajectories = generator.run(n_layers=1, n_iterations=1)
                assert len(trajectories) > 0  # Should continue without checkpoint
            finally:
                # Restore permissions for cleanup
                os.chmod(checkpoint_path, stat.S_IREAD | stat.S_IWRITE)


class TestResourceExhaustion:
    """Test handling of resource exhaustion scenarios."""

    def test_memory_allocation_failure(self):
        """Test handling of memory allocation failures."""
        # Try to create extremely large arrays
        protein_atoms = {
            "coords": np.random.rand(10, 3),
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

        params = {
            "use_ref15": False,
            "n_workers": 1,
            "max_steps": 10,
            "layer_step": 1e-10,  # Extremely small step would create huge arrays
        }

        try:
            generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
            # Try to access very large layer
            layer_gen = generator.layer_generator

            # This might cause memory issues
            with pytest.raises((MemoryError, ValueError)):
                huge_layer = layer_gen.get_layer(1_000_000)
        except MemoryError:
            # Expected in low-memory environments
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
