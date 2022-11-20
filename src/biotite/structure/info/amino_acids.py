# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller"
__all__ = ["amino_acid_names"]

import json
import numpy as np
from os.path import join, dirname, realpath


_info_dir = dirname(realpath(__file__))
# Data is taken from
# ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
# (2022/09/17)
# The json-file contains all three-letter-codes of the components where
# the data item `_chem_comp.type` is equal to one of the following
# values:
# D-PEPTIDE LINKING; D-PEPTIDE NH3 AMINO TERMINUS; 
# D-beta-peptide, C-gamma linking; D-gamma-peptide, C-delta linking; 
# D-peptide NH3 amino terminus; D-peptide linking; 
# L-PEPTIDE COOH CARBOXY TERMINUS; L-PEPTIDE LINKING; 
# L-beta-peptide, C-gamma linking; L-gamma-peptide, C-delta linking; 
# L-peptide COOH carboxy terminus; L-peptide NH3 amino terminus; 
# L-peptide linking; PEPTIDE LINKING; peptide linking
with open(join(_info_dir, "amino_acids.json"), "r") as file:
    _amino_acids = json.load(file)

def amino_acid_names():
    """
    Get a list of amino acid three-letter codes according to the PDB
    chemical compound dictionary.

    Returns
    -------
    amino_acid_names : list
        A list of three-letter-codes containing residues that are
        peptide monomers.
    """
    return _amino_acids
