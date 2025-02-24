import pytest
from biotite.interface.version import VersionError, requires_version


def test_requires_version_for_incompatible_version():
    """
    Expect an exception if the required package version for a function is not met.
    """

    @requires_version("biotite", ">999")
    def function_with_incompatible_version():
        pass

    with pytest.raises(VersionError):
        function_with_incompatible_version()


def test_requires_version_for_missing_package():
    """
    Expect an exception if the required package for a function is not installed.
    """
    with pytest.raises(ImportError):

        @requires_version("missing", ">=1.0")
        def _function_with_missing_package():
            pass
