"""Test-wide fixtures and pytest options."""

import pytest


def pytest_addoption(parser):
    """Register ``--device`` exactly once.

    Pytest discovers both this file **and** ``fluxmd/core/conftest.py``;
    the second registration would raise::

        ValueError: option names {'--device'} already added

    We scan ``parser._addedargs`` for an existing option whose ``dest`` is
    ``"device"`` and skip re-adding it.
    """

    added = getattr(parser, "_option_string_actions", {})
    already_registered = any(getattr(opt, "dest", None) == "device" for opt in added.values())

    if already_registered:
        return

    try:
        parser.addoption(
            "--device",
            action="store",
            choices=["cpu", "cuda"],
            default="cpu",
            help="Device on which to run FluxMD tests",
        )
    except ValueError:
        # Another conftest registered it first; ignore.
        pass


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
