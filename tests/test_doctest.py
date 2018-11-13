# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import pkgutil
import doctest
from importlib import import_module
import pytest


@pytest.mark.parametrize("package_name, context_package_name", [
    ("biotite",                      "biotite"                ),
    ("biotite.sequence",             "biotite.sequence"       ),
    ("biotite.sequence.align",       "biotite.sequence"       ),
    ("biotite.sequence.phylo",       "biotite.sequence"       ),
    ("biotite.sequence.graphics",    "biotite.sequence"       ),
    ("biotite.structure.io",         "biotite.sequence"       ),
    ("biotite.structure.io.fasta",   "biotite.sequence"       ),
    ("biotite.structure.io.genbank", "biotite.sequence"       ),
    ("biotite.structure",            "biotite.structure"      ),
    ("biotite.structure.io",         "biotite.structure"      ),
    ("biotite.structure.io.pdb",     "biotite.structure"      ),
    ("biotite.structure.io.pdbx",    "biotite.structure"      ),
    ("biotite.structure.io.npz",     "biotite.structure"      ),
    ("biotite.structure.io.mmtf",    "biotite.structure"      ),
    ("biotite.database.entrez",      "biotite.database.entrez"),
    ("biotite.database.rcsb",        "biotite.database.rcsb"  ),
    ("biotite.application",          "biotite.application"    ),
    ("biotite.application.blast",    "biotite.application"    ),
    ("biotite.application.muscle",   "biotite.application"    ),
    ("biotite.application.clustalo", "biotite.application"    ),
    ("biotite.application.mafft",    "biotite.application"    ),
    ("biotite.application.dssp",     "biotite.application"    ),
])
def test_doctest(package_name, context_package_name):
    """
    Run all doctest strings in all Biotite subpackages.
    """
    # Collect all attributes of this package and its subpackages
    # as globals for the doctests
    globs = {}
    context_package = import_module(context_package_name)
    mod_names = _list_modules(context_package, True)
    for modname in mod_names:
        mod = import_module(modname)
        attrs = mod.__all__
        globs.update({attr : getattr(mod, attr) for attr in attrs})

    # Run doctests
    package = import_module(package_name)
    mod_names = _list_modules(package, True)
    for modname in mod_names:
        mod = import_module(modname)
        results = doctest.testmod(
            mod, extraglobs=globs,
            verbose=False, report=False
        )
        try:
            assert results.failed == 0
        except AssertionError:
            print(f"Failing doctest in module {mod}")
            raise


def _list_modules(package, recursive):
    """
    Recursively list module names.
    """
    modnames = []
    for finder, modname, ispkg in pkgutil.walk_packages(package.__path__):
        abs_modname = f"{package.__name__}.{modname}"
        if ispkg:
            if recursive:
                subpackage = import_module(abs_modname)
                modnames.extend(_list_modules(subpackage, recursive))
        else:
            modnames.append(abs_modname)
    return modnames


test_doctest("biotite.sequence", "biotite.sequence")