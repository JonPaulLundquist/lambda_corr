#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 20:20:25 2026

@author: Jon Paul Lundquist
"""

# lambda_corr/__init__.py
from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "lambda_corr","lambda_corr_nb", "__version__",
]

# bind functions directly on the package
from .lambda_corr import lambda_corr, lambda_corr_nb

# Package version
try:
    __version__ = version("lambda_corr")
except PackageNotFoundError:
    __version__ = "0.0.0"