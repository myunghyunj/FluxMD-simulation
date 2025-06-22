"""FluxMD package initializer."""

from .__version__ import __version__


def get_version() -> str:
    """Return package version."""
    return __version__

__all__ = ['__version__', 'get_version']
