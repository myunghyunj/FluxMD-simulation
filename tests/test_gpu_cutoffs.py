import pytest

pytest.importorskip("torch")

from gpu_accelerated_flux import HierarchicalDistanceFilter


def test_cutoff_stages_last_distance():
    hdf = HierarchicalDistanceFilter(device='cpu')
    last_distance, _ = hdf.cutoff_stages[-1]
    assert last_distance == pytest.approx(4.5)
