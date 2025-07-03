from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="device to run tests on (cpu or gpu)",
    )


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


def pytest_ignore_collect(path, config):
    """Skip legacy preprint tests bundled with the repo."""
    return "preprint" in str(path)


# Explicitly ignore bundled legacy tests to avoid option clashes
collect_ignore = [
    str(Path(__file__).resolve().parents[1] / "preprint" / "FluxMD v 1.3.0" / "tests")
]
