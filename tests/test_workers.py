"""
Unit tests for CPU worker utilities
"""

import pytest
from unittest.mock import patch
from fluxmd.utils.cpu import parse_workers, get_optimal_workers, format_workers_info


class TestParseWorkers:
    """Test worker count parsing"""
    
    @patch('multiprocessing.cpu_count')
    def test_parse_workers_auto(self, mock_cpu_count):
        """Test auto-detection of workers"""
        mock_cpu_count.return_value = 8
        
        # Test various auto inputs
        assert parse_workers(None) == 7
        assert parse_workers("") == 7
        assert parse_workers("auto") == 7
        
    @patch('multiprocessing.cpu_count')
    def test_parse_workers_single_core(self, mock_cpu_count):
        """Test auto-detection on single core system"""
        mock_cpu_count.return_value = 1
        
        # Should still return at least 1
        assert parse_workers(None) == 1
        assert parse_workers("auto") == 1
        
    def test_parse_workers_explicit(self):
        """Test explicit worker counts"""
        assert parse_workers("1") == 1
        assert parse_workers("4") == 4
        assert parse_workers("16") == 16
        assert parse_workers(8) == 8
        
    def test_parse_workers_invalid(self):
        """Test invalid inputs"""
        with pytest.raises(ValueError, match="Invalid worker count"):
            parse_workers("invalid")
            
        with pytest.raises(ValueError, match="Invalid worker count"):
            parse_workers("abc")
            
        with pytest.raises(ValueError, match="Worker count must be >= 1"):
            parse_workers("0")
            
        with pytest.raises(ValueError, match="Worker count must be >= 1"):
            parse_workers("-1")
            
    @patch('multiprocessing.cpu_count')
    def test_get_optimal_workers(self, mock_cpu_count):
        """Test optimal worker detection"""
        mock_cpu_count.return_value = 8
        assert get_optimal_workers() == 7
        
        mock_cpu_count.return_value = 1
        assert get_optimal_workers() == 1
        
    @patch('multiprocessing.cpu_count')
    def test_format_workers_info(self, mock_cpu_count):
        """Test worker info formatting"""
        mock_cpu_count.return_value = 8
        
        # Serial processing
        info = format_workers_info(1)
        assert "1 worker (serial processing" in info
        assert "8 cores available" in info
        
        # Auto-detected
        info = format_workers_info(7)
        assert "7 workers (auto-detected" in info
        assert "8 cores total" in info
        
        # Custom count
        info = format_workers_info(4)
        assert "4 workers" in info
        assert "8 cores total" in info 