import pytest
import biotite.interface.pymol as pymol_interface
from biotite.interface.pymol.startup import get_and_set_pymol_instance


def test_get_and_set_pymol_instance():
    """
    Test :func:`get_and_set_pymol_instance()` with a provided *PyMOL*
    instance.
    It is not possible to test this with another instance, since this
    would require duplicate *PyMOL* launching.
    """
    assert pymol_interface.pymol is get_and_set_pymol_instance(pymol_interface.pymol)


def test_get_and_set_pymol_instance_typecheck():
    """
    Expect an exception, if :func:`get_and_set_pymol_instance()` is
    given a non-*PyMOL* instance as parameter.
    """
    with pytest.raises(pymol_interface.DuplicatePyMOLError):
        assert get_and_set_pymol_instance(42)
