"""
xpublish_wms is not a real package, just a set of best practices examples.
"""

from xpublish_edr.cf_wms_router import cf_wms_router

__all__ = ["cf_wms_router"]

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"