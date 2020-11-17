# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller"
__all__ = ["nucleotide_names"]

import json
import numpy as np
from os.path import join, dirname, realpath


_info_dir = dirname(realpath(__file__))
# Data is taken from
# ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
# (2020/10/21)
# The json-file contains all three-letter-codes of the components where
# the data item `_chem_comp.type` is equal to one of the following
# values:
# DNA OH 3 prime terminus, DNA OH 5 prime terminus, DNA linking,
# RNA OH 3 prime terminus, RNA OH 5 prime terminus, RNA linking,
# L-RNA LINKING, L-DNA LINKING
with open(join(_info_dir, "nucleotides.json"), "r") as file:
    _nucleotides = json.load(file)

def nucleotide_names():
    """
    Get a list of nucleotide three-letter codes according to the PDB
    chemical compound dictionary.

    Returns
    -------
    nucleotide_names : list
        A list of three-letter-codes containing residues that are
        DNA/RNA-Linking.
    """
    return _nucleotides
