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
