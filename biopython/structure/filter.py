# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
This module provides utility functions for creating filters on atom arrays and atom array stacks.
"""

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack

_ext_aa_list = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
                "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
                "MSE", "ASX", "GLX", "SEC"]

def filter_amino_acids(array):
    return ( np.in1d(array.res_name, _ext_aa_list) & (array.res_id != -1) )

def filter_backbone(array):
    return ( ((array.atom_name == "N") |
              (array.atom_name == "CA") |
              (array.atom_name == "C")) &
              filter_amino_acids(array) )