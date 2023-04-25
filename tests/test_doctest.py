# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import pkgutil
import doctest
from os.path import join
import tempfile
from importlib import import_module
import numpy as np
import pytest
import biotite.structure.io as strucio
from .util import is_not_installed, cannot_import, cannot_connect_to


NCBI_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/"
RCSB_URL = "https://www.rcsb.org/"
UNIPROT_URL = "https://www.uniprot.org/"
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/"


@pytest.mark.parametrize("package_name, context_package_names", [
    pytest.param(
        "biotite",
        []
    ),
    pytest.param(
        "biotite.sequence",
        []
    ),
    pytest.param(
        "biotite.sequence.align",
        ["biotite.sequence"]
    ),
    pytest.param(
        "biotite.sequence.phylo",
        ["biotite.sequence"]
    ),
    pytest.param(
        "biotite.sequence.graphics",
        ["biotite.sequence"],
        marks = pytest.mark.skipif(
            cannot_import("matplotlib"), reason="Matplotlib is not installed"
        )
    ),
    pytest.param(
        "biotite.sequence.io",
        ["biotite.sequence"]
    ),
    pytest.param(
        "biotite.sequence.io.fasta",
        ["biotite.sequence"]
    ),
    pytest.param(
        "biotite.sequence.io.fastq",
        ["biotite.sequence"]
    ),
    pytest.param(
        "biotite.sequence.io.genbank",
        ["biotite.sequence", "biotite.database.entrez"],
        marks = pytest.mark.skipif(
            cannot_connect_to(NCBI_URL), reason="NCBI Entrez is not available"
        )
    ),
    pytest.param(
        "biotite.sequence.io.gff",
        ["biotite.sequence", "biotite.sequence.io.fasta"],
        marks = pytest.mark.filterwarnings("ignore:")
    ),
    pytest.param(
        "biotite.structure",
        ["biotite.structure.io", "biotite.structure.info"]
    ),
    pytest.param(
        "biotite.structure.graphics",
        ["biotite.structure"],    
        marks = pytest.mark.skipif(
            cannot_import("matplotlib"), reason="Matplotlib is not installed"
        ),
    ),
    pytest.param(
        "biotite.structure.io",
        ["biotite.structure"]
    ),
    pytest.param(
        "biotite.structure.io.pdb",
        ["biotite.structure", "biotite"]
    ),
    pytest.param(
        "biotite.structure.io.pdbx",
        ["biotite.structure"]
    ),
    pytest.param(
        "biotite.structure.io.pdbqt",
        ["biotite.structure", "biotite.structure.info"]
    ),
    pytest.param(
        "biotite.structure.io.npz",
        ["biotite.structure"]
    ),
    pytest.param(
        "biotite.structure.io.mmtf",
        ["biotite.structure"]
    ),
    pytest.param(
        "biotite.structure.io.mol",
        ["biotite.structure"]
    ),
    pytest.param(
        "biotite.structure.info",
        ["biotite.structure"]
    ),
    pytest.param(
        "biotite.database.entrez",
        [],                           
        marks = pytest.mark.skipif(
            cannot_connect_to(NCBI_URL), reason="NCBI Entrez is not available"
        )
    ),
    pytest.param(
        "biotite.database.rcsb",
        [],
        marks = pytest.mark.skipif(
            cannot_connect_to(RCSB_URL), reason="RCSB PDB is not available"
        )
    ),
    pytest.param(
        "biotite.database.uniprot",
        [],
        marks = pytest.mark.skipif(
            cannot_connect_to(UNIPROT_URL), reason="UniProt is not available"
        )
    ),
    pytest.param(
        "biotite.database.pubchem",
        ["biotite.structure.info"],
        marks = pytest.mark.skipif(
            cannot_connect_to(PUBCHEM_URL), reason="PubChem is not available"
        )
    ),
    pytest.param(
        "biotite.application",
        ["biotite.application.clustalo", "biotite.sequence"],            
        marks = pytest.mark.skipif(
            is_not_installed("clustalo"), reason="Software is not installed"
        )
    ),
    pytest.param(
        "biotite.application.blast",
        [],
    ),
    # Do not test Muscle due to version clash
    #pytest.param(
    #    "biotite.application.muscle",
    #    ["biotite.sequence"],
    #    marks = pytest.mark.skipif(
    #        is_not_installed("muscle"), reason="Software is not installed")
    #    ),
    pytest.param(
        "biotite.application.clustalo",
        ["biotite.sequence"],
        marks = pytest.mark.skipif(
            is_not_installed("clustalo"), reason="Software is not installed"
        )
    ),
    pytest.param(
        "biotite.application.mafft",
        ["biotite.sequence"],
        marks = pytest.mark.skipif(
            is_not_installed("mafft"), reason="Software is not installed")
        ),
    pytest.param(
        "biotite.application.sra", ["biotite.sequence"],
        marks = pytest.mark.skipif(
            is_not_installed("fasterq-dump"),
            reason="Software is not installed"
        )
    ),
    pytest.param(
        "biotite.application.tantan",
        ["biotite.sequence"],
        marks = pytest.mark.skipif(
            is_not_installed("tantan"), reason="Software is not installed")
        ),
    pytest.param(
        "biotite.application.viennarna",
        ["biotite.sequence"],
        marks = pytest.mark.skipif(
            is_not_installed("RNAfold") | is_not_installed("RNAplot"),
            reason="Software is not installed"
        )
    ),                                      
    pytest.param(
        "biotite.application.dssp",
        ["biotite.structure"],
        marks = pytest.mark.skipif(
            is_not_installed("mkdssp"), reason="Software is not installed"
        )
    ),
    pytest.param(
        "biotite.application.autodock",
        ["biotite.structure", "biotite.structure.info"],
        marks = pytest.mark.skipif(
            is_not_installed("vina"), reason="Software is not installed"
        )
    ),
])
def test_doctest(package_name, context_package_names):
    """
    Run all doctest strings in all Biotite subpackages.
    """
    # Collect all attributes of this package and its subpackages
    # as globals for the doctests
    globs = {}
    #The package itself is also used as context
    for name in context_package_names + [package_name]:
        context_package = import_module(name)
        globs.update(
            {attr : getattr(context_package, attr)
             for attr in dir(context_package)}
        )
    
    # Add fixed names for certain paths
    globs["path_to_directory"]  = tempfile.gettempdir()
    globs["path_to_structures"] = join(".", "tests", "structure", "data")
    globs["path_to_sequences"]  = join(".", "tests", "sequence", "data")
    # Add frequently used modules
    globs["np"] = np
    # Add frequently used objects
    globs["atom_array_stack"] = strucio.load_structure(
        join(".", "tests", "structure", "data", "1l2y.mmtf"),
        include_bonds=True
    )
    globs["atom_array"] = globs["atom_array_stack"][0]
    
    # Adjust NumPy print formatting
    np.set_printoptions(precision=3, floatmode="maxprec_equal")

    # Run doctests
    # This test does not use 'testfile()' or 'testmod()'
    # due to problems with doctest identification for Cython modules
    # More information below
    package = import_module(package_name)
    runner = doctest.DocTestRunner(
        verbose = False,
        optionflags = 
            doctest.ELLIPSIS |
            doctest.REPORT_ONLY_FIRST_FAILURE |
            doctest.NORMALIZE_WHITESPACE
    )
    for test in doctest.DocTestFinder(exclude_empty=False).find(
        package, package.__name__,
        # It is necessary to set 'module' to 'False', as otherwise
        # Cython functions and classes would be falsely identified
        # as members of an external module by 'DocTestFinder._find()'
        # and consequently would be ignored
        #
        # Setting 'module=False' omits this check
        # This check is not necessary as the biotite subpackages
        # ('__init__.py' modules) should only contain attributes, that
        # are part of the package itself.
        module=False,
        extraglobs=globs
    ):
        runner.run(test)
    results = doctest.TestResults(runner.failures, runner.tries)
    try:
        assert results.failed == 0
    except AssertionError:
        print(f"Failing doctest in module {package}")
        raise