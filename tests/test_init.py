# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Daniel Bauer"

import biotite
import pytest


def test_version_number():
    version = biotite.__version__
    assert hasattr(biotite, "__version__")