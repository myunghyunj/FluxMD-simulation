"""Version information for FluxMD"""

__version__ = "0.1.0"
__version_info__ = tuple(int(i) for i in __version__.split("."))
__author__ = "FluxMD Contributors"
__email__ = "fluxmd@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024 FluxMD Contributors"
__url__ = "https://github.com/yourusername/FluxMD"

def get_version():
    """Return the current version string"""
    return __version__

def get_version_tuple():
    """Return the current version as a tuple of integers"""
    return __version_info__