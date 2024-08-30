import pytest
from biotite.structure.info.ccd import get_ccd


@pytest.fixture(autouse=True, scope="session")
def load_ccd():
    """
    Ensure that the CCD is already loaded to avoid biasing tests with its loading time.
    """
    get_ccd()
