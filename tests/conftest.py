# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np

def pytest_sessionstart(session):
    """
    Compile Cython source files, if Cython is installed and files are
    not compiled, yet.
    """
    try:
        import pyximport
        pyximport.install(
            build_in_temp=False,
            language_level=3
        )
    except ImportError:
        pass