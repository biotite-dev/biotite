from importlib.metadata import version
import biotite


def test_version():
    """
    Check if version imported from version.py is correct.
    """
    assert biotite.__version__ == version("biotite")