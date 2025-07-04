"""
Comprehensive physics validation tests for Matryoshka trajectory engine.

These tests ensure physical correctness and numerical stability of:
- Energy conservation
- Diffusion coefficients  
- Layer hopping probabilities
- Force calculations
- Numerical stability over long trajectories
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch

from fluxmd.core.matryoshka_generator import MatryoshkaTrajectoryGenerator
from fluxmd.core.dynamics.brownian_roller import BrownianSurfaceRoller
from fluxmd.core.surface.ses_builder import SESBuilder, SurfaceMesh
from fluxmd.core.ref15_energy import REF15EnergyCalculator, AtomContext


class TestEnergyConservation:
    """Test energy conservation in Brownian dynamics."""
    
    def test_total_energy_bounded(self):
        """Verify total energy remains bounded over long trajectories."""
        # Create minimal test system
        protein_atoms = {
            'coords': np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]]),
            'names': np.array(['CA', 'CA', 'CA']),
            'radii': np.ones(3) * 1.8,
            'masses': np.ones(3) * 12.0,
            'resnames': np.array(['ALA', 'ALA', 'ALA'])
        }
        
        ligand_atoms = {
            'coords': np.array([[10, 0, 0]]),
            'names': np.array(['C']),
            'masses': np.array([12.0])
        }
        
        # Create simple surface
        vertices = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0]])
        faces = np.array([[0, 1, 2]])
        surface = SurfaceMesh(vertices, faces)
        
        # Track energies
        energies = []
        
        def mock_energy_calc(pos, quat, layer):
            # Simple harmonic potential
            dist = np.linalg.norm(pos)
            energy = 0.5 * dist**2
            energies.append(energy)
            return energy
        
        # Create roller with energy tracking
        ligand_sphere = {
            'radius': 1.0,
            'mass': 12.0,
            'center': np.array([10, 0, 0]),
            'inertia': 2.0
        }
        
        roller = BrownianSurfaceRoller(
            surface=surface,
            ligand_sphere=ligand_sphere,
            anchors=(np.array([0, 0, 0]), np.array([5, 5, 0])),
            T=298.15,
            k_surf=1.0,
            k_guid=0.5,
            energy_calculator=mock_energy_calc,
            seed=42
        )
        
        # Run short trajectory
        trajectory = roller.run(max_steps=1000)
        
        # Check energy bounds
        energies = np.array(energies)
        assert len(energies) > 0, "No energies recorded"
        assert np.all(np.isfinite(energies)), "Non-finite energies detected"
        
        # Energy should remain bounded (not explode)
        max_energy = np.max(energies)
        mean_energy = np.mean(energies)
        assert max_energy < 10 * mean_energy, f"Energy explosion detected: max={max_energy}, mean={mean_energy}"
        
    def test_diffusion_coefficient_accuracy(self):
        """Verify diffusion coefficients match theoretical values."""
        # Test system parameters
        T = 298.15  # K
        viscosity = 0.00089  # Pa·s (water)
        radius = 5.0  # Å
        mass = 100.0  # amu
        
        # Create minimal ligand sphere
        ligand_sphere = {
            'radius': radius,
            'mass': mass,
            'center': np.array([0, 0, 0]),
            'inertia': (2.0/5.0) * mass * radius**2
        }
        
        # Create roller to access diffusion calculations
        surface = SurfaceMesh(np.array([[0, 0, 0]]), np.array([]))
        roller = BrownianSurfaceRoller(
            surface=surface,
            ligand_sphere=ligand_sphere,
            anchors=(np.array([0, 0, 0]), np.array([10, 0, 0])),
            T=T,
            viscosity=viscosity
        )
        
        # Check calculated diffusion coefficients
        D_t, D_r = roller.D_t, roller.D_r
        
        # Theoretical values (with unit conversions)
        KB_KCAL = 0.0019872041  # kcal/mol/K
        kT = KB_KCAL * T
        eta_converted = viscosity * 1.439e-4  # Conversion factor to kcal·ps/mol/Å²
        
        D_t_theory = kT / (6 * np.pi * eta_converted * radius)
        D_r_theory = kT / (8 * np.pi * eta_converted * radius**3)
        
        # Check within 5% (accounting for approximations)
        assert abs(D_t - D_t_theory) / D_t_theory < 0.05, f"D_t mismatch: {D_t} vs {D_t_theory}"
        assert abs(D_r - D_r_theory) / D_r_theory < 0.05, f"D_r mismatch: {D_r} vs {D_r_theory}"


class TestLayerHopping:
    """Test layer hopping mechanics and probabilities."""
    
    def test_metropolis_criterion(self):
        """Verify layer hopping follows correct Metropolis criterion."""
        from fluxmd.core.surface.layer_stream import MatryoshkaLayerGenerator
        
        # Create test surfaces
        base_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        base_faces = np.array([[0, 1, 2]])
        base_surface = SurfaceMesh(base_vertices, base_faces)
        
        layer_gen = MatryoshkaLayerGenerator(base_surface, step=1.0)
        
        # Mock energy calculator
        energy_calls = []
        def mock_energy(pos, quat, layer):
            # Energy increases with layer (uphill)
            energy = -10.0 + layer * 2.0
            energy_calls.append((pos.copy(), layer, energy))
            return energy
        
        # Create roller
        ligand_sphere = {
            'radius': 1.0,
            'mass': 12.0,
            'center': np.array([0.5, 0.5, 1.0]),
            'inertia': 2.0
        }
        
        roller = BrownianSurfaceRoller(
            surface=base_surface,
            ligand_sphere=ligand_sphere,
            anchors=(np.array([0, 0, 0]), np.array([1, 1, 0])),
            layer_generator=layer_gen,
            current_layer_idx=1,
            energy_calculator=mock_energy,
            hop_probability=1.0,  # Always attempt
            T=298.15,
            seed=42
        )
        
        # Test hop attempts
        position = np.array([0.5, 0.5, 2.0])
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Attempt multiple hops to get statistics
        hop_results = []
        for i in range(100):
            roller.rng = np.random.default_rng(i)  # Different seeds
            result = roller._attempt_layer_hop(position, quaternion)
            hop_results.append(result)
        
        # Should have some accepted and rejected hops
        n_accepted = sum(hop_results)
        assert 0 < n_accepted < 100, f"Unrealistic hop acceptance: {n_accepted}/100"
        
    def test_layer_boundary_conditions(self):
        """Test behavior at layer boundaries."""
        # Create layer generator
        base_surface = SurfaceMesh(
            np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 2]])
        )
        layer_gen = MatryoshkaLayerGenerator(base_surface, step=1.0)
        
        # Test negative layer access
        with pytest.raises(Exception):
            layer_gen.get_layer(-1)
            
        # Test very large layer
        large_layer = layer_gen.get_layer(100)
        assert large_layer is not None
        assert len(large_layer.vertices) == len(base_surface.vertices)


class TestZeroVolumeHandling:
    """Test handling of zero atomic volumes in REF15."""
    
    def test_zero_volume_solvation(self):
        """Verify solvation energy calculation handles zero volumes gracefully."""
        calculator = REF15EnergyCalculator()
        
        # Create atoms with zero volume
        atom1 = AtomContext(
            atom_type='H',  # Often has zero volume
            coords=np.array([0, 0, 0]),
            formal_charge=0.0
        )
        
        atom2 = AtomContext(
            atom_type='C',
            coords=np.array([2, 0, 0]),
            formal_charge=0.0
        )
        
        # Mock zero volume parameters
        with patch.object(calculator.params, 'get_lk_params') as mock_lk:
            # Return zero volume for hydrogen
            def side_effect(atom_type):
                if atom_type == 'H':
                    return (0.5, 1.0, 0.0)  # dgfree, lambda, volume=0
                else:
                    return (1.0, 1.5, 10.0)  # normal values
            
            mock_lk.side_effect = side_effect
            
            # Calculate energy - should not crash
            with pytest.warns(RuntimeWarning, match="Zero atomic volume"):
                energy = calculator._calculate_solvation_energy(atom1, atom2, 2.0)
            
            assert np.isfinite(energy), f"Non-finite energy: {energy}"
            
    def test_ref15_energy_bounds(self):
        """Test that REF15 energies remain bounded."""
        calculator = REF15EnergyCalculator()
        
        # Create test atoms
        atom1 = AtomContext(
            atom_type='C',
            coords=np.array([0, 0, 0]),
            formal_charge=0.0
        )
        
        atom2 = AtomContext(
            atom_type='N',
            coords=np.array([1, 0, 0]),
            formal_charge=-1.0
        )
        
        # Test at various distances
        distances = [0.5, 1.0, 2.0, 4.0, 6.0, 10.0]
        energies = []
        
        for dist in distances:
            atom2.coords = np.array([dist, 0, 0])
            energy = calculator.calculate_interaction_energy(atom1, atom2, dist)
            energies.append(energy)
            
        energies = np.array(energies)
        
        # Check bounds
        assert np.all(np.isfinite(energies)), f"Non-finite energies: {energies}"
        assert np.all(np.abs(energies) < 1000), f"Unrealistic energies: {energies}"


class TestNumericalStability:
    """Test numerical stability over long simulations."""
    
    def test_quaternion_normalization(self):
        """Verify quaternion remains normalized during integration."""
        from fluxmd.core.dynamics.brownian_roller import quaternion_multiply
        
        # Start with normalized quaternion
        q = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Apply many small rotations
        for i in range(10000):
            # Small random rotation
            angle = 0.001
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            
            half_angle = angle / 2
            dq = np.array([
                np.cos(half_angle),
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle)
            ])
            
            q = quaternion_multiply(dq, q)
            
            # Check normalization every 100 steps
            if i % 100 == 0:
                norm = np.linalg.norm(q)
                assert abs(norm - 1.0) < 1e-10, f"Quaternion drift at step {i}: |q|={norm}"
                
    def test_force_calculation_stability(self):
        """Test force calculations remain stable at edge cases."""
        # Create surface
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        faces = np.array([[0, 1, 2]])
        surface = SurfaceMesh(vertices, faces)
        
        ligand_sphere = {
            'radius': 1.0,
            'mass': 12.0,
            'center': np.array([0, 0, 0]),
            'inertia': 2.0
        }
        
        roller = BrownianSurfaceRoller(
            surface=surface,
            ligand_sphere=ligand_sphere,
            anchors=(np.array([0, 0, 0]), np.array([1, 1, 0])),
            k_surf=2.0
        )
        
        # Test forces at various positions
        test_positions = [
            np.array([0.5, 0.5, 0.001]),  # Very close to surface
            np.array([0.5, 0.5, 10.0]),    # Far from surface
            np.array([0, 0, 1.0]),         # At vertex
            np.array([10, 10, 1.0]),       # Far from surface laterally
        ]
        
        for pos in test_positions:
            force = roller._surface_force(pos, target_distance=1.0)
            assert np.all(np.isfinite(force)), f"Non-finite force at {pos}: {force}"
            assert np.linalg.norm(force) < 100, f"Unrealistic force at {pos}: {force}"


class TestMemoryLeaks:
    """Test for memory leaks in layer generation."""
    
    def test_layer_cache_management(self):
        """Verify layer cache doesn't grow unbounded."""
        base_surface = SurfaceMesh(
            np.random.rand(100, 3) * 10,
            np.random.randint(0, 100, (200, 3))
        )
        
        layer_gen = MatryoshkaLayerGenerator(base_surface, step=1.0)
        
        # Access layers in non-sequential pattern
        access_pattern = [0, 5, 2, 8, 1, 10, 3, 15, 0, 20]
        
        for layer_idx in access_pattern:
            layer = layer_gen.get_layer(layer_idx)
            # Cache should only keep adjacent layers
            assert len(layer_gen._cache) <= 3, f"Cache too large: {len(layer_gen._cache)} entries"
            
    def test_trajectory_memory_usage(self):
        """Test memory usage doesn't grow excessively during long trajectories."""
        protein_atoms = {
            'coords': np.random.rand(50, 3) * 20,
            'names': np.array(['CA'] * 50),
            'radii': np.ones(50) * 1.8,
            'masses': np.ones(50) * 12.0,
            'resnames': np.array(['ALA'] * 50)
        }
        
        ligand_atoms = {
            'coords': np.random.rand(5, 3),
            'names': np.array(['C'] * 5),
            'masses': np.ones(5) * 12.0
        }
        
        params = {
            'use_ref15': False,
            'n_workers': 1,
            'max_steps': 100  # Short trajectory
        }
        
        generator = MatryoshkaTrajectoryGenerator(protein_atoms, ligand_atoms, params)
        
        # Run trajectory and check memory
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        trajectories = generator.run(n_layers=1, n_iterations=1)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        # Should not use excessive memory for small system
        assert mem_increase < 100, f"Excessive memory usage: {mem_increase:.1f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])