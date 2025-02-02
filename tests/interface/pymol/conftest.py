import pytest
import biotite.interface.pymol as pymol_interface


@pytest.fixture(autouse=True)
def reset_pymol():
    pymol_interface.reset()
