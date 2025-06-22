"""
Unit tests for REF15 zero volume handling
"""

import pytest
import warnings
import numpy as np
from unittest.mock import Mock, patch

from fluxmd.core.ref15_energy import REF15EnergyCalculator, AtomContext


class TestREF15ZeroVolume:
    """Test zero volume handling in REF15 energy calculations"""
    
    def test_zero_volume_warning(self):
        """Test that zero volume triggers warning but doesn't crash"""
        calc = REF15EnergyCalculator()
        
        # Create mock atom contexts
        atom1 = AtomContext(
            atom_type="C", 
            coords=np.array([0.0, 0.0, 0.0]),
            neighbor_count=0
        )
        atom2 = AtomContext(
            atom_type="N",
            coords=np.array([3.0, 0.0, 0.0]),
            neighbor_count=0
        )
        
        # Mock the parameter getter to return zero volume
        with patch.object(calc.params, 'get_lk_params') as mock_get_lk:
            # First call (atom1): normal volume
            # Second call (atom2): zero volume
            mock_get_lk.side_effect = [
                (1.0, 3.5, 10.0),  # dgfree, lambda, volume for atom1
                (1.0, 3.5, 0.0)    # dgfree, lambda, ZERO volume for atom2
            ]
            
            # Should warn but not crash
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                energy = calc._calculate_solvation_energy(atom1, atom2, 3.0)
                
                # Check that warning was issued
                assert len(w) == 1
                assert issubclass(w[0].category, RuntimeWarning)
                assert "Zero atomic volume detected" in str(w[0].message)
                assert "fallback 1.0 Å³" in str(w[0].message)
                
                # Should return a finite number, not NaN or inf
                assert np.isfinite(energy)
    
    def test_both_zero_volumes(self):
        """Test handling when both atoms have zero volume"""
        calc = REF15EnergyCalculator()
        
        atom1 = AtomContext(
            atom_type="X", 
            coords=np.array([0.0, 0.0, 0.0]),
            neighbor_count=0
        )
        atom2 = AtomContext(
            atom_type="Y",
            coords=np.array([3.0, 0.0, 0.0]),
            neighbor_count=0
        )
        
        # Mock both atoms to have zero volume
        with patch.object(calc.params, 'get_lk_params') as mock_get_lk:
            mock_get_lk.side_effect = [
                (1.0, 3.5, 0.0),   # Both atoms have zero volume
                (1.0, 3.5, 0.0)
            ]
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                energy = calc._calculate_solvation_energy(atom1, atom2, 3.0)
                
                # Should warn and handle gracefully
                assert len(w) == 1
                assert "Zero atomic volume detected" in str(w[0].message)
                assert np.isfinite(energy)
    
    def test_normal_volumes_no_warning(self):
        """Test that normal volumes don't trigger warnings"""
        calc = REF15EnergyCalculator()
        
        atom1 = AtomContext(
            atom_type="C", 
            coords=np.array([0.0, 0.0, 0.0]),
            neighbor_count=0
        )
        atom2 = AtomContext(
            atom_type="N",
            coords=np.array([3.0, 0.0, 0.0]),
            neighbor_count=0
        )
        
        # Mock normal volumes
        with patch.object(calc.params, 'get_lk_params') as mock_get_lk:
            mock_get_lk.side_effect = [
                (1.0, 3.5, 10.0),  # Normal volumes
                (1.0, 3.5, 15.0)
            ]
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                energy = calc._calculate_solvation_energy(atom1, atom2, 3.0)
                
                # Should not warn
                volume_warnings = [warning for warning in w 
                                 if "Zero atomic volume" in str(warning.message)]
                assert len(volume_warnings) == 0
                assert np.isfinite(energy)
    
    def test_parameter_validation_logging(self):
        """Test that parameter validation logs zero volume types"""
        
        # Mock params with some zero volumes
        mock_params = Mock()
        mock_params.lk_params = ['C', 'N', 'O', 'ZERO1', 'ZERO2']
        
        def mock_get_lk_params(atom_type):
            if atom_type.startswith('ZERO'):
                return 1.0, 3.5, 0.0  # Zero volume
            else:
                return 1.0, 3.5, 10.0  # Normal volume
        
        mock_params.get_lk_params = mock_get_lk_params
        
        with patch('fluxmd.core.ref15_energy.get_ref15_params', return_value=mock_params):
            with patch('fluxmd.core.ref15_energy.logger') as mock_logger:
                # This should trigger validation warnings
                calc = REF15EnergyCalculator()
                
                # Check that logger.warning was called
                assert mock_logger.warning.call_count >= 1
                
                # Check warning content
                warning_calls = [call for call in mock_logger.warning.call_args_list]
                volume_warnings = [call for call in warning_calls 
                                 if 'zero volume' in str(call).lower()]
                assert len(volume_warnings) > 0 