# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
A subpackage for handling molecular structures.

In this context an atom is described by two kinds of attributes: the
coordinates and the annotations. The annotations include information
about polypetide chain id, residue id, residue name, hetero atom
information, atom name and optionally more. The coordinates are a
`NumPy` float :class:`ndarray` of length 3, containing the x, y and z
coordinates.

An :class:`Atom` contains data for a single atom, it stores the
annotations as scalar values and the coordinates as length 3
:class:`ndarray`.

An :class:`AtomArray` stores data for an entire structure model
containing *n* atoms.
Therefore the annotations are represented as :class:`ndarray` objects of
length *n*, the so called annotation arrays.
The coordinates are a *(n x 3)* :class:`ndarray`.

An :class:`AtomArrayStack` stores data for *m* models, where each model
contains the same atoms at different positions.
Hence, the annotation arrays are represented as :class:`ndarray` objects
of length *n* like the :class:`AtomArray`, while the coordinates are a
*(m x n x 3)* :class:`ndarray`.

Like an :class:`AtomArray` can be iterated to obtain :class:`Atom`
objects, an :class:`AtomArrayStack` yields :class:`AtomArray` objects.
All three types must not be subclassed.

The following annotation categories are mandatory:

=========  ===========  =================  =======================================
Category   Type         Examples           Description
=========  ===========  =================  =======================================
chain_id   string (U4)  'A','S','AB', ...  Polypeptide chain
res_id     int          1,2,3, ...         Sequence position of residue
ins_code   string (U1)  '', 'A','B',..     PDB insertion code (iCode)
res_name   string (U5)  'GLY','ALA', ...   Residue name
hetero     bool         True, False        False for ``ATOM``, true for ``HETATM``
atom_name  string (U6)  'CA','N', ...      Atom name
element    string (U2)  'C','O','SE', ...  Chemical Element
=========  ===========  =================  =======================================

For all :class:`Atom`, :class:`AtomArray` and :class:`AtomArrayStack`
objects these annotations are initially set with default values.
Additionally to these annotations, an arbitrary amount of annotation
categories can be added via :func:`add_annotation()` or
:func:`set_annotation()`.
The annotation arrays can be accessed either via the method
:func:`get_annotation()` or directly (e.g. ``array.res_id``).

The following annotation categories are optionally used by some
functions:

=========  ===========  =================   =========================================
Category   Type         Examples            Description
=========  ===========  =================   =========================================
atom_id    int          1,2,3, ...          Atom serial number
b_factor   float        0.9, 12.3, ...      Temperature factor
occupancy  float        .1, .3, .9, ...     Occupancy
charge     int          -2,-1,0,1,2, ...    Electric charge of the atom
sym_id     string       '1','2','3', ...    Symmetry ID for assemblies/symmetry mates
=========  ===========  =================   =========================================

For each type, the attributes can be accessed directly.
Both :class:`AtomArray` and :class:`AtomArrayStack` support
*NumPy* style indexing.
The index is propagated to each attribute.
If a single integer is used as index,
an object with one dimension less is returned
(:class:`AtomArrayStack` -> :class:`AtomArray`,
:class:`AtomArray` -> :class:`Atom`).
If a slice, index array or a boolean mask is given, a substructure is
returned
(:class:`AtomArrayStack` -> :class:`AtomArrayStack`,
:class:`AtomArray` -> :class:`AtomArray`)
As in *NumPy*, these are not necessarily deep copies of the originals:
The attributes of the sliced object may still point to the original
:class:`ndarray`.
Use the :func:`copy()` method if a deep copy is required.

Bond information can be associated to an :class:`AtomArray` or
:class:`AtomArrayStack` by setting the ``bonds`` attribute with a
:class:`BondList`.
A :class:`BondList` specifies the indices of atoms that form chemical
bonds.
Some functionalities require that the input structure has an associated
:class:`BondList`.
If no :class:`BondList` is associated, the ``bonds`` attribute is
``None``.

Based on the implementation in *NumPy* arrays, this package furthermore
contains a comprehensive set of functions for structure analysis,
manipulation and visualization.

The universal length unit in this package is Å.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"

from .atoms import *
from .basepairs import *
from .bonds import *
from .box import *
from .celllist import *
from .chains import *
from .charges import *
from .compare import *
from .density import *
from .dotbracket import *
from .error import *
from .filter import *
from .geometry import *
from .hbond import *
from .integrity import *
from .mechanics import *
from .molecules import *
from .pseudoknots import *
from .rdf import *
from .repair import *
from .residues import *
from .rings import *
from .sasa import *
from .sequence import *
from .sse import *
from .superimpose import *
from .tm import *
from .transform import *
# util and segments are used internally
