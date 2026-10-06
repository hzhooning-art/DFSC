"""Opt-in DLL setup for the local hybrid Python/Anaconda test environment.

This file is imported by Python only when this fixtures directory is
explicitly on PYTHONPATH. It leaves normal and frozen environments alone.
"""

from __future__ import annotations

import os


_dll_directory = os.environ.get("P4_SCIPY_DLL_DIR")
if _dll_directory and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_dll_directory)
