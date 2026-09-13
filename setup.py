"""
The build is configured in ``pyproject.toml``.
This file only exists to install a Rust toolchain into a temporary directory,
if none is available, so that the Rust extension can also be built on platforms
without prebuilt wheels.
"""

import os
import shutil
from setuptools import setup

if not shutil.which("cargo"):
    from puccinialin import setup_rust

    # The returned environment variables point `setuptools-rust` to the
    # temporary toolchain
    os.environ.update(setup_rust())

setup()
