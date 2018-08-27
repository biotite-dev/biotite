# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for handling molecular structures. 

In this context an atom is described by two kinds of attributes: the
coordinates and the annotations. The annotations include information
about polypetide chain id, residue id, residue name, hetero atom
information, atom name and optionally more. The coordinates are a
`NumPy` float `ndarray` of length 3, containing the x, y and z
coordinates.

An `Atom` contains data for a single atom, it stores the annotations as
scalar values and the coordinates as length 3 `ndarray`.
An `AtomArray` stores data for an entire structure model containing *n*
atoms. Therefore the annotations are represented as `ndarray`s of
length *n*, so called annotation arrays. The coordinates are a (n x 3)
`ndarray` .
`AtomArrayStack` stores data for *m* models. Each `AtomArray` in
the `AtomArrayStack` has the same annotation arrays, since each atom
must be represented in all models in the stack. Each model may differ in
atom coordinates. Therefore the annotation arrays are represented as
`ndarrays` of length *n*, while the coordinates are a (m x n x 3)
`ndarray` .
All types must not be subclassed.

The following annotation categories are mandatory:

=========  ===========  =================  =============================
Category   Type         Examples           Description
=========  ===========  =================  =============================
chain_id   string (U3)  'A','S','AB', ...  Polypeptide chain
res_id     int          1,2,3, ...         Sequence position of residue
res_name   string (U3)  'GLY','ALA', ...   Residue name
hetero     bool         True, False        True for non AA/NUC residues
atom_name  string (U6)  'CA','N', ...      Atom name
element    string (U2)  'C','O','SE', ...  Chemical Element
=========  ===========  =================  =============================

For all `Atom`, `AtomArray` and `AtomArrayStack` objects these
annotations must be set, otherwise some functions will not work or
errors will occur.
Additionally to these annotations, an arbitrary amount of annotation
categories can be added (use `add_annotation()` or `add_annotation()`
for this).
The annotation arrays can be accessed either via the function
`get_annotation()` or directly (e.g. ``array.res_id``).

The following annotation categories are optionally used by some
functions:

=========  ===========  =================   ============================
Category   Type         Examples            Description
=========  ===========  =================   ============================
atom_id    int          1,2,3, ...          Atom serial number
b_factor   float        0.9, 12.3, ...      Temperature factor
occupancy  float        .1, .3, .9, ...     Occupancy
charge     int          -2,-1,0,1,2, ...    Electric charge of the atom
=========  ===========  =================   ============================

For each type, the attributes can be accessed directly. Both `AtomArray`
and `AtomArrayStack` support `NumPy` style indexing, the index is
propagated to each attribute. If a single integer is used as index,
an object with one dimension less is returned
(`AtomArrayStack` -> `AtomArray`, `AtomArray` -> `Atom`).
Do not expect a deep copy, when sclicing an `AtomArray` or
`AtomArrayStack`. The attributes of the sliced object may still point
to the original `ndarray`.

An optional attribute for `AtomArray` and `AtomArrayStack` instances
are associated `BondList` objects, that specifiy the indices of atoms
that form a chemical bonds.

Based on the implementation in `NumPy` arrays, this package furthermore
contains functions for structure analysis, manipulation and
visualisation.
"""

__author__ = "Patrick Kunzmann"

from .atoms import *
from .bonds import *
from .celllist import *
from .compare import *
from .error import *
from .filter import *
from .geometry import *
from .hbond import *
from .integrity import *
from .mechanics import *
from .residues import *
from .sasa import *
from .sse import *
from .superimpose import *
from .transform import *
# util is used internally
