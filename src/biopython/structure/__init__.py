# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

"""
A subpackage for handling protein structures. 

This subpackage enables efficient and easy handling of protein structure data
by representation of atom attributes in `numpy` arrays. These atom attributes
include polypetide chain id, residue id, residue name, hetero residue
information, atom name and atom coordinates.

The package contains mainly three types: `Atom`, `AtomArray` and
`AtomArrayStack`. An `Atom` contains data for a single atom, an `AtomArray`
stores data for an entire model and `AtomArrayStack` stores data for multiple
models, where each model differs in the atom coordinates. Both, `AtomArrray` and
`AtomArrayStack`, store the attributes in `numpy` arrays. This approach has
multiple advantages:
    
    - Convenient selection of atoms in a structure
      by using `numpy` style indexing
    - Fast calculations on structures using C-accelerated `ndarray` operations
    - Simple implementation of custom calculations
    
Based ony the implementation in `numpy` arrays, this package furthermore
contains functions for structure analysis, manipulation and visualisation.
An `Entity` from `BIO.PDB` can be converted into an `AtomArrray` or
`AtomArrayStack` and vice versa.
"""

from .adjacency import *
from .atoms import *
from .compare import *
from .error import *
from .filter import *
from .geometry import *
from .integrity import *
from .residues import *
from .sasa import *
from .superimpose import *
from .transform import *
# util is used internal
from .vis import *