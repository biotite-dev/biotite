# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

# Enable runtime type checking of Biotite annotations during the test session
# The hook must be installed before `biotite` is imported for the first time
# (which happens in other `conftest.py` modules), so it lives at module top level rather
# than inside `pytest_sessionstart`.
from beartype.claw import beartype_packages

beartype_packages(("biotite",))


def pytest_sessionstart(session):
    """
    Compile Cython source files, if Cython is installed and files are
    not compiled, yet.
    """
    try:
        import numpy as np
        import pyximport

        pyximport.install(
            build_in_temp=False,
            setup_args={"include_dirs": np.get_include()},
            language_level=3,
        )
    except ImportError:
        pass
