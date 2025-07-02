"""Test-wide fixtures and pytest options."""

import pytest


def pytest_addoption(parser):
    """Register a ``--device`` option once across all conftests."""

    added = getattr(parser, "_option_string_actions", {})
    already_registered = any(
        getattr(opt, "dest", None) == "device" for opt in added.values()
    )

    if not already_registered:
        try:
            parser.addoption(
                "--device",
                action="store",
                choices=["cpu", "cuda"],
                default="cpu",
                help="Device on which to run FluxMD tests",
            )
        except ValueError:
            # Another conftest registered the option in the meantime.
            pass


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
