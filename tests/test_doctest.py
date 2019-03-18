# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import pkgutil
import doctest
import os.path
import numpy as np
from importlib import import_module
import pytest
import biotite
import biotite.structure.io as strucio


@pytest.mark.parametrize("package_name, context_package_names", [
    pytest.param("biotite",                      []                          ),
    pytest.param("biotite.sequence",             []                          ),
    pytest.param("biotite.sequence.align",       ["biotite.sequence"]        ),
    pytest.param("biotite.sequence.phylo",       ["biotite.sequence"]        ),
    pytest.param("biotite.sequence.graphics",    ["biotite.sequence"],
                 marks=pytest.mark.xfail(raises=ImportError)                 ),
    pytest.param("biotite.sequence.io",          ["biotite.sequence"]        ),
    pytest.param("biotite.sequence.io.fasta",    ["biotite.sequence"]        ),
    pytest.param("biotite.sequence.io.fastq",    ["biotite.sequence"]        ),
    pytest.param("biotite.sequence.io.genbank",  ["biotite.sequence",
                                                  "biotite.database.entrez"] ),
    pytest.param("biotite.structure",            ["biotite.structure.io"]    ),
    pytest.param("biotite.structure.io",         ["biotite.structure"]       ),
    pytest.param("biotite.structure.io.pdb",     ["biotite.structure"]       ),
    pytest.param("biotite.structure.io.pdbx",    ["biotite.structure"]       ),
    pytest.param("biotite.structure.io.npz",     ["biotite.structure"]       ),
    pytest.param("biotite.structure.io.mmtf",    ["biotite.structure"]       ),
    pytest.param("biotite.structure.info",       ["biotite.structure"]       ),
    pytest.param("biotite.database.entrez",      []                          ),
    pytest.param("biotite.database.rcsb",        []                          ),
    pytest.param("biotite.application",          []                          ),
    pytest.param("biotite.application.blast",    [],
                 marks=pytest.mark.xfail(raises=OSError)                 ),
    pytest.param("biotite.application.muscle",   [],
                 marks=pytest.mark.xfail(raises=OSError)                 ),
    pytest.param("biotite.application.clustalo", [],
                 marks=pytest.mark.xfail(raises=OSError)                 ),
    pytest.param("biotite.application.mafft",    [],
                 marks=pytest.mark.xfail(raises=OSError)                 ),
    pytest.param("biotite.application.dssp",     [],
                 marks=pytest.mark.xfail(raises=OSError)                 ),
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
    globs["np"] = np
    # Add frequently used objects
    globs["atom_array_stack"] = strucio.load_structure(
        "./tests/structure/data/1l2y.mmtf"
    )
    globs["atom_array"] = globs["atom_array_stack"][0]
    # Adjust NumPy print formatting
    np.set_printoptions(precision=3, floatmode="maxprec_equal")

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
    for finder, modname, ispkg in pkgutil.iter_modules(package.__path__):
        abs_modname = f"{package.__name__}.{modname}"
        if ispkg:
            if recursive:
                subpackage = import_module(abs_modname)
                modnames.extend(_list_modules(subpackage, recursive))
        else:
            modnames.append(abs_modname)
    return modnames