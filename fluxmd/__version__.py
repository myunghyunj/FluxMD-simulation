"""Version information for FluxMD"""

__version__ = "1.4.0b1"
__version_info__ = (1, 4, 0, "b1")
__author__ = "Myunghyun Jeong"
__email__ = "mhjonathan@gm.gist.ac.kr"
__license__ = "MIT"
__copyright__ = "Copyright 2025 FluxMD"
__url__ = "https://github.com/myunghyunj/FluxMD"

def get_version():
    """Return the current version string"""
    return __version__

def get_version_tuple():
    """Return the current version as a tuple of integers"""
    return __version_info__
