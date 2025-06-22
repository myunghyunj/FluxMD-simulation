import pytest
from unittest.mock import patch
from fluxmd.utils.cpu import parse_workers

@patch('multiprocessing.cpu_count', return_value=8)
def test_auto_none(mock_cpu):
    assert parse_workers(None)==7
    assert parse_workers('')==7
    assert parse_workers('auto')==7

@patch('multiprocessing.cpu_count', return_value=1)
def test_single_core(mock_cpu):
    assert parse_workers(None)==1

def test_explicit():
    assert parse_workers('3')==3
    with pytest.raises(ValueError):
        parse_workers('zero') 