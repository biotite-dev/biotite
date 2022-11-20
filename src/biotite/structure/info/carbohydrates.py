# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Tom David Müller"
__all__ = ["carbohydrate_names"]

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
# D-SACCHARIDE; D-saccharide; D-saccharide, alpha linking; 
# D-saccharide, beta linking; L-SACCHARIDE; L-saccharide; 
# L-saccharide, alpha linking; L-saccharide, beta linking; SACCHARIDE; 
# saccharide
with open(join(_info_dir, "carbohydrates.json"), "r") as file:
    _carbohydrates = json.load(file)

def carbohydrate_names():
    """
    Get a list of carbohydrate three-letter codes according to the PDB
    chemical compound dictionary.

    Returns
    -------
    carbohydrate_names : list
        A list of three-letter-codes containing residues that are
        saccharide monomers.
    """
    return _carbohydrates
