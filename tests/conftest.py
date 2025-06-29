import pytest


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
