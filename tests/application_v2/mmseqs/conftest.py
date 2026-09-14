# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from pathlib import Path
import pytest
from biotite.application_v2.mmseqs import FoldseekApp, MMseqsApp
from tests.util import is_not_installed


@pytest.fixture(scope="session")
def mmseqs_app():
    if is_not_installed("mmseqs"):
        pytest.skip("MMseqs2 is not installed")
    return MMseqsApp()


@pytest.fixture(scope="session")
def foldseek_app():
    if is_not_installed("foldseek"):
        pytest.skip("Foldseek is not installed")
    return FoldseekApp()


@pytest.fixture(scope="session")
def structure_path():
    return Path("tests/structure/data/pdb/1aki.cif")


@pytest.fixture(scope="session")
def multimer_path():
    """
    A structure comprising more than one chain.
    """
    return Path("tests/structure/data/pdb/1k6p.cif")
