import pytest
import biotite.application as application


def test_deprecated_alias():
    """
    The deprecated `biotite.application_v2` package warns and forwards to
    `biotite.application`, including its subpackages.
    """
    with pytest.warns(DeprecationWarning):
        import biotite.application_v2 as application_v2
    assert application_v2.LocalApp is application.LocalApp
    assert application_v2.VersionError is application.VersionError

    import biotite.application.viennarna as viennarna
    import biotite.application_v2.viennarna as viennarna_v2
    from biotite.application_v2.dssp import DsspApp

    assert DsspApp is application.dssp.DsspApp
    assert viennarna_v2 is viennarna
