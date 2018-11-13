# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import pkgutil
import doctest
import os.path
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module
import pytest
import biotite
import biotite.structure.io as strucio


@pytest.mark.parametrize("package_name, context_package_names", [
    ("biotite",                      []                          ),
    ("biotite.sequence",             []                          ),
    ("biotite.sequence.align",       ["biotite.sequence"]        ),
    ("biotite.sequence.phylo",       ["biotite.sequence"]        ),
    ("biotite.sequence.graphics",    ["biotite.sequence"]        ),
    ("biotite.sequence.io",          ["biotite.sequence"]        ),
    ("biotite.sequence.io.fasta",    ["biotite.sequence"]        ),
    ("biotite.sequence.io.genbank",  ["biotite.sequence",
                                      "biotite.database.entrez"] ),
    ("biotite.structure",            []                          ),
    ("biotite.structure.io",         ["biotite.structure"]       ),
    ("biotite.structure.io.pdb",     ["biotite.structure"]       ),
    ("biotite.structure.io.pdbx",    ["biotite.structure"]       ),
    ("biotite.structure.io.npz",     ["biotite.structure"]       ),
    ("biotite.structure.io.mmtf",    ["biotite.structure"]       ),
    ("biotite.database.entrez",      []                          ),
    ("biotite.database.rcsb",        []                          ),
    ("biotite.application",          []                          ),
    ("biotite.application.blast",    []                          ),
    ("biotite.application.muscle",   []                          ),
    ("biotite.application.clustalo", []                          ),
    ("biotite.application.mafft",    []                          ),
    ("biotite.application.dssp",     []                          ),
])
def test_doctest(package_name, context_package_names):
    """
    Run all doctest strings in all Biotite subpackages.
    """
    # Collect all attributes of this package and its subpackages
    # as globals for the doctests
    globs = {}
    mod_names = []
    #The package itself is also used as context
    for name in context_package_names + [package_name]:
        context_package = import_module(name)
        mod_names += _list_modules(context_package, False)
    for modname in mod_names:
        mod = import_module(modname)
        attrs = mod.__all__
        globs.update({attr : getattr(mod, attr) for attr in attrs})
    # Add fixed names for certain paths
    globs["path_to_directory"]  = biotite.temp_dir()
    globs["path_to_structures"] = "./tests/structure/data/"
    globs["path_to_sequences"]  = "./tests/sequence/data/"
    # Add frequently used modules
    globs["np"]  = np
    globs["plt"] = plt
    # Add frequently used objects
    globs["atom_array_stack"] = strucio.load_structure(
        "./tests/structure/data/1l2y.mmtf"
    )
    globs["atom_array"] = globs["atom_array_stack"][0]

    # Run doctests
    package = import_module(package_name)
    mod_names = _list_modules(package, False)
    for modname in mod_names:
        mod = import_module(modname)
        results = doctest.testmod(
            mod, extraglobs=globs,
            optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
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