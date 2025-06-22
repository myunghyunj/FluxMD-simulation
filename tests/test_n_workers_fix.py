"""Test for n_workers parameter handling in MatryoshkaTrajectoryGenerator."""

import pytest
import numpy as np
from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.utils.cpu import parse_workers


def create_minimal_test_data():
    """Create minimal test data for MatryoshkaTrajectoryGenerator."""
    protein_atoms = {
        'coords': np.random.rand(10, 3) * 20,
        'names': np.array(['CA'] * 10),
        'radii': np.ones(10) * 1.8,
        'masses': np.ones(10) * 12.0,
        'resnames': np.array(['ALA'] * 10)
    }
    
    ligand_atoms = {
        'coords': np.random.rand(5, 3),
        'names': np.array(['C'] * 5),
        'masses': np.ones(5) * 12.0,
    }
    
    return protein_atoms, ligand_atoms


class TestNWorkersHandling:
    """Test n_workers parameter handling."""
    
    def test_n_workers_none(self):
        """Test that n_workers=None is handled correctly."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': None, 'use_ref15': False}
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers is not None
        assert generator.n_workers >= 1
        assert isinstance(generator.n_workers, int)
    
    def test_n_workers_empty_string(self):
        """Test that n_workers='' is handled correctly."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': '', 'use_ref15': False}
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers is not None
        assert generator.n_workers >= 1
        assert isinstance(generator.n_workers, int)
    
    def test_n_workers_auto(self):
        """Test that n_workers='auto' is handled correctly."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': 'auto', 'use_ref15': False}
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers is not None
        assert generator.n_workers >= 1
        assert isinstance(generator.n_workers, int)
    
    def test_n_workers_integer(self):
        """Test that integer n_workers is preserved."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': 4, 'use_ref15': False}
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers == 4
    
    def test_n_workers_string_integer(self):
        """Test that string integer n_workers is parsed correctly."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': '3', 'use_ref15': False}
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers == 3
    
    def test_n_workers_missing(self):
        """Test that missing n_workers parameter is handled correctly."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'use_ref15': False}  # n_workers not in params
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers is not None
        assert generator.n_workers >= 1
        assert isinstance(generator.n_workers, int)
    
    def test_n_workers_invalid(self):
        """Test that invalid n_workers values are handled gracefully."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': 'invalid', 'use_ref15': False}
        
        # Should handle gracefully with warning and auto-detection
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        assert generator.n_workers is not None
        assert generator.n_workers >= 1
        assert isinstance(generator.n_workers, int)
    
    def test_n_workers_in_run_method(self):
        """Test that n_workers is still valid when run() is called."""
        protein_atoms, ligand_atoms = create_minimal_test_data()
        params = {'n_workers': None, 'use_ref15': False, 'checkpoint_dir': None}
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        # The run method should handle None n_workers gracefully
        # We'll test with minimal iterations to keep it fast
        trajectories = generator.run(n_layers=1, n_iterations=1)
        
        assert generator.n_workers is not None
        assert len(trajectories) >= 0  # May be 0 if checkpointed


def test_parse_workers_function():
    """Test the parse_workers utility function directly."""
    # Test various inputs
    assert parse_workers(None) >= 1
    assert parse_workers('') >= 1
    assert parse_workers('auto') >= 1
    assert parse_workers(4) == 4
    assert parse_workers('4') == 4
    
    # Test invalid inputs
    with pytest.raises(ValueError):
        parse_workers('invalid')
    
    with pytest.raises(ValueError):
        parse_workers(0)
    
    with pytest.raises(ValueError):
        parse_workers(-1)