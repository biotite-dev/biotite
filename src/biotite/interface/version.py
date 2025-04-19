# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.interface"
__author__ = "Patrick Kunzmann"
__all__ = ["VersionError", "requires_version"]


import functools
import importlib.metadata
from packaging.specifiers import SpecifierSet
from packaging.version import Version

# Stores the variant of interface functions
# compatible with the respective installed package version
_functions_for_version = {}


class VersionError(Exception):
    """
    This exception is raised when the installed version of an interfaced package is
    incompatible with all implemented variants of a function.
    """

    pass


def require_package(package):
    """
    Check if the given package is installed and raise an exception if not.

    Parameters
    ----------
    package : str
        The name of the package to be checked.

    Raises
    ------
    ImportError
        If the package is not installed.

    Notes
    -----
    It is useful to call this function in the ``__init__.py`` of each ``interface``
    subpackage, to obtain clear error messages about missing dependencies.
    """
    if importlib.util.find_spec(package) is None:
        raise ImportError(f"'{package}' is not installed")


def requires_version(package, version_specifier):
    """
    Declare a function variant that is compatible with a specific version range of the
    interfaced package.

    Parameters
    ----------
    package : str
        The name of the interfaced package.
    version_specifier : str or list of str
        The :pep:`440` version specifier(s) for the interfaced package that are
        compatible with the function.
        Multiple constraints can be either given as a list of strings or as a single
        comma-separated string.
    """

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            function_for_version = _functions_for_version.get(function.__name__)
            if function_for_version is None:
                raise VersionError(
                    f"No variant of '{function.__name__}()' "
                    f"found for installed '{package}'=={package_version}'"
                )
            return function_for_version(*args, **kwargs)

        if isinstance(version_specifier, str):
            specifier = SpecifierSet(version_specifier)
        else:
            specifier = SpecifierSet.intersection(*version_specifier)
        try:
            package_version = Version(importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            raise ImportError(
                f"'{function.__name__}()' requires the '{package}' package"
            )
        if package_version in specifier:
            _functions_for_version[function.__name__] = function

        return wrapper

    return decorator
