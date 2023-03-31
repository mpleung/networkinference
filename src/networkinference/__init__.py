"""
networkinference

This package provides methods for inference with dependent (especially network-dependent) data.
"""
__version__ = '0.0.3'

from .inference import IPW, OLS, TSLS
from .core import core
