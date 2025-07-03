"""Test-wide fixtures and pytest options."""

import pytest


def pytest_addoption(parser):
    """Register ``--device`` exactly once.

    *Pytest* discovers both this file **and**
    ``fluxmd/core/conftest.py``; the second registration of the same
    flag raises::

        ValueError: option names {'--device'} already added

    We scan ``parser._addedargs`` for an existing option whose
    ``dest`` equals ``"device"`` and silently skip re-adding it.
    """

    added = getattr(parser, "_addedargs", {})
    already_registered = any(
        getattr(opt, "dest", None) == "device"
        for opt in (added.values() if isinstance(added, dict) else added)
    )

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
        # Another plugin raced us—ignore duplicate registration.
        pass


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
