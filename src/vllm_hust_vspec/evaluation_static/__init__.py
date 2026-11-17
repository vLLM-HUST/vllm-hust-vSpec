"""Installable, standalone native evaluation worker assets.

The native vLLM child loads modules in this directory by their top-level names
from PYTHONPATH. It must not import this package or the vSpec plugin.
"""
