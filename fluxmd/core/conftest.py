import pytest


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--device",
            action="store",
            choices=["cpu", "cuda"],
            default="cpu",
            help="Device on which to run FluxMD simulations",
        )
    except ValueError:
        # already registered by another plugin
        pass

