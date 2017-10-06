import biopython
import pytest


def test_version_number():
    version = biopython.__version__
    assert hasattr(biopython, "__version__")