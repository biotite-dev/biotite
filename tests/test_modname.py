# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pkgutil
from os.path import dirname, join, isdir, splitext
import importlib
import pytest
from .util import cannot_import


def find_all_modules(package_name, src_dir):
    """
    Recursively look for Python modules in the given source directory.
    (Sub-)Packages are not considered as modules.
    """
    module_names = []
    for _, module_name, is_package in pkgutil.iter_modules([src_dir]):
        full_module_name = f"{package_name}.{module_name}"
        if is_package:
            module_names.extend(find_all_modules(
                full_module_name,
                join(src_dir, module_name)
            ))
        else:
            module_names.append(full_module_name)
    return module_names


@pytest.mark.skipif(
    cannot_import("matplotlib") | cannot_import("mdtraj"),
    reason="Optional dependencies are not met"
)
@pytest.mark.parametrize(
    "module_name",
    find_all_modules(
        "biotite",
        join(dirname(dirname(__file__)), "src", "biotite")
    )
)
def test_module_name(module_name):
    """
    Test whether the '__name__' attribute of each module in Biotite is
    set to the name of the subpackage.
    This needs to be tested, since by default the '__name__' attribute
    is equal to the name of the module.
    For example, we expect 'biotite.structure' instead of
    'biotite.structure.atoms'.
    """
    # Remove the part after the last '.' of the module name
    # to obtain the package name
    package_name = ".".join(module_name.split(".")[:-1])
    module = importlib.import_module(module_name)
    assert module.__name__ == package_name