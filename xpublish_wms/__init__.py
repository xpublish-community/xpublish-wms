"""
xpublish_wms is not a real package, just a set of best practices examples.
"""

from .plugin import CfWmsPlugin

__all__ = ["CfWmsPlugin"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"
