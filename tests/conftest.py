import sys
from pathlib import Path

import pytest

# Ensure the repository root is on sys.path so tests can import fluxmd without
# requiring an editable install. This matches the lightweight CI setup that only
# installs test tooling.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


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
