"""
Going 3D - The Structure subpackage
===================================

.. currentmodule:: biotite.structure
   
:mod:`biotite.structure` is a *Biotite* subpackage for handling
molecular structures.
This subpackage enables efficient and easy handling of protein structure
data by representing atom attributes in *NumPy* :class:`ndarray`
objects.
These atom attributes include so called *annotations*
(polypetide chain id, residue id, residue name, hetero residue
information, atom name, element, etc.)
and the atom coordinates.

The package contains three central types: :class:`Atom`,
:class:`AtomArray` and :class:`AtomArrayStack`.
An :class:`Atom` contains data for a single atom, an :class:`AtomArray`
stores data for an entire model and :class:`AtomArrayStack` stores data
for multiple models, where each model contains the same atoms but
differs in the atom coordinates.
Both, :class:`AtomArray` and :class:`AtomArrayStack`, store the
attributes in `NumPy` arrays. This approach has multiple advantages:
    
    - Convenient selection of atoms in a structure
      by using *NumPy* style indexing
    - Fast calculations on structures using C-accelerated
      :class:`ndarray` operations
    - Simple implementation of custom calculations
    
Based on the implementation using :class:`ndarray` objects, this package
also contains functions for structure analysis and manipulation.

.. Note::
    The universal length unit in *Biotite* is Å.
    This includes coordinates, distances, surface areas, etc.

Creating structures
-------------------

Let's begin by constructing some atoms:
"""

import biotite.structure as struc

atom1 = struc.Atom([0,0,0], chain_id="A", res_id=1, res_name="GLY",
                   atom_name="N", element="N")
atom2 = struc.Atom([0,1,1], chain_id="A", res_id=1, res_name="GLY",
                   atom_name="CA", element="C")
atom3 = struc.Atom([0,0,2], chain_id="A", res_id=1, res_name="GLY",
                   atom_name="C", element="C")

########################################################################
# The first parameter are the coordinates (internally converted into an
# :class:`ndarray`), the other parameters are annotations.
# The annotations shown in this example are mandatory:
# The chain ID, residue ID, residue name, insertion code, atom name,
# element and whether the atom is not in protein/nucleotide chain
# (*hetero*).
# If you miss one of these, they will get a default value.
# The mandatory annotation categories are originated in *ATOM* and
# *HETATM* records in the PDB format.
# The description of each annotation can be viewed in the
# :doc:`API reference </apidoc/biotite.structure>`.
# Additionally, you can specify an arbitrary amount of custom
# annotations, like B-factors, charge, etc.
# In most cases you won't work with :class:`Atom` instances and in even
# fewer cases :class:`Atom` instances are created as it is done in the
# above example.
# 
# If you want to work with an entire molecular structure, containing an
# arbitrary amount of atoms, you have to use so called atom arrays.
# An atom array can be seen as an array of atom instances
# (hence the name).
# But instead of storing :class:`Atom` instances in a list, an
# :class:`AtomArray` instance contains one :class:`ndarray` for each
# annotation and the coordinates.
# In order to see this in action, we first have to create an array from
# the atoms we constructed before.
# Then we can access the annotations and coordinates of the atom array
# simply by specifying the attribute.

import numpy as np

array = struc.array([atom1, atom2, atom3])
print("Chain ID:", array.chain_id)
print("Residue ID:", array.res_id)
print("Atom name:", array.atom_name)
print("Coordinates:", array.coord)
print()
print(array)

########################################################################
# The :func:`array()` builder function takes any iterable object
# containing :class:`Atom` instances.
# If you wanted to, you could even use another :class:`AtomArray`, which
# functions also as an iterable object of :class:`Atom` objects.
# An alternative way of constructing an array would be creating an
# :class:`AtomArray` by using its constructor, which fills the
# annotation arrays and coordinates with the type respective *zero*
# value.
# In our example all annotation arrays have a length of 3, since we used
# 3 atoms to create it.
# A structure containing *n* atoms is represented by annotation arrays
# of length *n* and coordinates of shape *(n,3)*.
# As the annotations and coordinates are simply :class:`ndarray`
# objects, they can be edited in the same manner.

array.chain_id[:] = "B"
array.coord[array.element == "C", 0] = 42
# It is also possible to replace an entire annotation with another array
array.res_id = np.array([1,2,3])
print(array)

########################################################################
# Apart from the structure manipulation functions we see later on, this
# is the usual way to edit structures in *Biotite*.
#
# .. warning:: For editing an annotation, the index must be applied to
#    the annotation and not to the :class:`AtomArray`, e.g.
#    ``array.chain_id[...] = "B"`` instead of
#    ``array[...].chain_id = "B"``.
#    The latter example is incorrect, as it creates a subarray of the
#    initial :class:`AtomArray` (discussed later) and then tries to
#    replace the annotation array with the new value.
# 
# If you want to add further annotation categories to an array, you have
# to call the :func:`add_annotation()` or :func:`set_annotation()`
# method at first. After that you can access the new annotation array
# like any other annotation array.

array.add_annotation("foo", dtype=bool)
array.set_annotation("bar", [1, 2, 3])
print(array.foo)
print(array.bar)

########################################################################
# In some cases, you might need to handle structures, where each atom is
# present in multiple locations
# (multiple models in NMR structures, MD trajectories).
# For the cases :class:`AtomArrayStack` objects are used, which
# represent a list of atom arrays.
# Since the atoms are the same for each frame, but only the coordinates
# change, the annotation arrays in stacks are still the same length *n*
# :class:`ndarray` objects as in atom arrays.
# However, a stack stores the coordinates in a *(m,n,3)*-shaped
# :class:`ndarray`, where *m* is the number of frames.
# A stack is constructed with :func:`stack()` analogous to the code
# snipped above.
# It is crucial that all arrays that should be stacked
# have the same annotation arrays, otherwise an exception is raised.
# For simplicity reasons, we create a stack containing two identical
# models, derived from the previous example.

stack = struc.stack([array, array.copy()])
print(stack)

########################################################################
# Loading structures from file
# ----------------------------
# 
# Usually structures are not built from scratch, but they are read from
# a file.
# Probably the most popular structure file format is the *PDB* format.
# For our purpose, we will work on a protein structure as small as
# possible, namely the miniprotein *TC5b* (PDB: ``1L2Y``).
# The structure of this 20-residue protein (304 atoms) has been
# elucidated via NMR.
# Thus, the corresponding PDB file consists of multiple (namely 38)
# models, each showing another conformation.
# 
# .. currentmodule:: biotite.structure.io.pdb
#
# At first we load the structure from a PDB file via the class
# :class:`PDBFile` in the subpackage :mod:`biotite.structure.io.pdb`.

from tempfile import gettempdir, NamedTemporaryFile
import biotite.structure.io.pdb as pdb
import biotite.database.rcsb as rcsb

pdb_file_path = rcsb.fetch("1l2y", "pdb", gettempdir())
pdb_file = pdb.PDBFile.read(pdb_file_path)
tc5b = pdb_file.get_structure()
print(type(tc5b).__name__)
print(tc5b.stack_depth())
print(tc5b.array_length())
print(tc5b.shape)

########################################################################
# The method :func:`PDBFile.get_structure()` returns an atom array stack
# unless the :obj:`model` parameter is specified,
# even if the file contains only one model.
# Alternatively, the module level function :func:`get_structure()`
# can be used.
# The following example
# shows how to write an atom array or stack back into a PDB file:

pdb_file = pdb.PDBFile()
pdb_file.set_structure(tc5b)
temp_file = NamedTemporaryFile(suffix=".pdb", delete=False)
pdb_file.write(temp_file.name)
temp_file.close()

########################################################################
# Other information (authors, secondary structure, etc.) cannot be
# extracted from PDB files, yet.
# This is a good place to mention, that it is recommended to use the
# modern PDBx/mmCIF format in favor of the PDB format.
# It solves limitations of the PDB format, that arise from the column
# restrictions.
# Furthermore, much more additional information is stored in these
# files.
# 
# .. currentmodule:: biotite.structure.io.pdbx
# 
# In contrast to PDB files, *Biotite* can read the entire content of
# PDBx/mmCIF files, which can be accessed in a dictionary like manner.
# At first, we read the file similarily to before, but this time we
# use the :class:`PDBxFile` class.

import biotite.structure.io.pdbx as pdbx

cif_file_path = rcsb.fetch("1l2y", "cif", gettempdir())
cif_file = pdbx.PDBxFile.read(cif_file_path)

########################################################################
# Now we can access the data like a dictionary of dictionaries.

print(cif_file["1L2Y", "audit_author"]["name"])

########################################################################
# The first index contains the data block and the category name.
# The data block could be omitted, since there is only one block in the
# file.
# This returns a dictionary.
# If the category is in a ``loop_`` category, i.e. the category's fields
# have a list of values, like in this case, the dictionary contains
# :class:`ndarray` objects of type string as values, otherwise the
# dictionary contains strings directly.
# The second index specifies the name of the subcategory, which is used
# as key in this dictionary and returns the corresponding
# :class:`ndarray`.
# Setting/adding a category in the file is done in a similar way:

cif_file["audit_author"] = {
    "name" : ["Doe, Jane", "Doe, John"],
    "pdbx_ordinal" : ["1","2"]
}

########################################################################
# In most applications only the structure itself
# (stored in the *atom_site* category) is relevant.
# :func:`get_structure()` and :func:`set_structure()` are convenience
# functions that are used to convert the
# ``atom_site`` category into an atom array (stack) and vice versa.

tc5b = pdbx.get_structure(cif_file)
# Do some fancy stuff
pdbx.set_structure(cif_file, tc5b)

########################################################################
# :func:`get_structure()` creates automatically an
# :class:`AtomArrayStack`, even if the file actually contains only a
# single model.
# If you would like to have an :class:`AtomArray` instead, you have to
# specifiy the :obj:`model` parameter.
# 
# .. currentmodule:: biotite.structure.io.mmtf
#
# If you want to parse a large batch of structure files or you have to
# load very large structure files, the usage of PDB or mmCIF files might
# be too slow for your requirements.
# In this case you probably might want to use MMTF files.
# MMTF files describe structures just like PDB and mmCIF files,
# but they are binary!
# This circumstance increases the downloading and parsing speed by
# several multiples.
# The usage is similar to :class:`PDBxFile`: The :class:`MMTFFile` class
# decodes the file and makes it raw information accessible.
# Via :func:`get_structure()` the data can be loaded into an atom array
# (stack) and :func:`set_structure()` is used to save it back into a
# MMTF file.

import numpy as np
import biotite.structure.io.mmtf as mmtf

mmtf_file_path = rcsb.fetch("1l2y", "mmtf", gettempdir())
mmtf_file = mmtf.MMTFFile.read(mmtf_file_path)
stack = mmtf.get_structure(mmtf_file)
array = mmtf.get_structure(mmtf_file, model=1)
# Do some fancy stuff
mmtf.set_structure(mmtf_file, array)

########################################################################
# A more low level access to MMTF files is also possible:
# An MMTF file is structured as dictionary, with each key being a
# structural feature like the coordinates, the residue ID or the
# secondary structure.
# Most of the fields are encoded to reduce to size of the file,
# but the whole decoding process is handled automatically by
# the :class:`MMTFFile` class:
# If a field is encoded the decoded
# :class:`ndarray` is returned, otherwise the value is directly
# returned.
# A list of all MMTF fields (keys) can be found in the
# `specification <https://github.com/rcsb/mmtf/blob/master/spec.md#fields>`_.
# The implementation of :class:`MMTFFile` decodes the encoded fields
# only when you need them, so no computation time is wasted on fields
# you are not interested in.

# Field is not encoded
print(mmtf_file["title"])
# Field is encoded and is automatically decoded
print(mmtf_file["groupIdList"])

########################################################################
# Setting fields of an MMTF file works in an analogous way for values,
# that should not be encoded.
# The situation is a little more complex for arrays, that should be
# encoded:
# Since arbitrarily named fields can be set in the file,
# :class:`MMTFFile` does not know which codec to use for encoding
# your array.
# Hence, you need to use the :func:`MMTFFile.set_array()` function.

mmtf_file["title"] = "Some other title"
print(mmtf_file["title"])
# Determine appropriate codec from the codec used originally
mmtf_file.set_array(
    "groupIdList",
    np.arange(20,40),
    codec=mmtf_file.get_codec("groupIdList"))
print(mmtf_file["groupIdList"])

########################################################################
# .. currentmodule:: biotite.structure.io.npz
# 
# An alternative file format for storing and loading atom arrays and
# stacks even faster, is the *NPZ* format.
# The big disadvantage is that the format is *Biotite*-exclusive:
# No other software will be able to read these files.
# These are simple binary files, that are used to store *NumPy* arrays.
# In case of atom arrays and stacks, the annotation arrays and
# coordinates are written/read to/from *npz* files via the
# :class:`NpzFile` class.
# Since no expensive data conversion has to be performed,
# this format is the fastest way to save and load atom arrays and
# stacks.

import biotite.structure.io.npz as npz

file = npz.NpzFile()
file.set_structure(array)
reloaded_array = file.get_structure()

########################################################################
# There are also some other supported file formats.
# For a full list, have a look at :mod:`biotite.structure.io`.
# 
# .. currentmodule:: biotite.structure.io
# 
# Since programmers are usually lazy and do not want to write more code
# than necessary, there are two convenient function for loading and
# saving atom arrays or stacks, unifying the forementioned file formats:
# :func:`load_structure()` takes a file path and outputs an array
# (or stack, if the file contains multiple models).
# Internally, this function uses the appropriate :class:`File` class,
# depending on the file format.
# The analogous :func:`save_structure()` function provides a shortcut
# for writing to structure files.
# The desired file format is inferred from the the extension of the
# provided file name.

import biotite.structure.io as strucio

stack_from_pdb = strucio.load_structure(pdb_file_path)
stack_from_cif = strucio.load_structure(cif_file_path)
temp_file = NamedTemporaryFile(suffix=".cif", delete=False)
strucio.save_structure(temp_file.name, stack_from_pdb)
temp_file.close()

########################################################################
# Reading trajectory files
# ^^^^^^^^^^^^^^^^^^^^^^^^
# 
# If the package *MDtraj* is installed, *Biotite* provides a read/write
# interface for different trajectory file formats.
# All supported trajectory formats have in common, that they store
# only coordinates.
# These can be extracted as :class:`ndarray` with the
# :func:`get_coord()` method.

from tempfile import NamedTemporaryFile
import requests
import biotite.structure.io.xtc as xtc

# Download 1L2Y as XTC file for demonstration purposes
temp_xtc_file = NamedTemporaryFile("wb", suffix=".xtc", delete=False)
response = requests.get(
    "https://raw.githubusercontent.com/biotite-dev/biotite/master/"
    "tests/structure/data/1l2y.xtc"
)
temp_xtc_file.write(response.content)

traj_file = xtc.XTCFile.read(temp_xtc_file.name)
coord = traj_file.get_coord()
print(coord.shape)

########################################################################
# If only an excerpt of frames is desired, the behavior of the
# :func:`read()` function can be customized with the `start`, `stop` and
# `step` parameters.

# Read only every second frame
traj_file = xtc.XTCFile.read(temp_xtc_file.name, step=2)
coord = traj_file.get_coord()
print(coord.shape)

########################################################################
# In order to extract an entire structure, i.e. an
# :class:`AtomArrayStack`, from a trajectory file, a *template*
# structure must be given, since the trajectory file contains only
# coordinate information.

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

mmtf_file_path = rcsb.fetch("1l2y", "mmtf", gettempdir())
mmtf_file = mmtf.MMTFFile.read(mmtf_file_path)
template = mmtf.get_structure(mmtf_file, model=1)

traj_file = xtc.XTCFile.read(temp_xtc_file.name)
trajectory = traj_file.get_structure(template)

temp_xtc_file.close()

########################################################################
# Array indexing and filtering
# ----------------------------
# 
# .. currentmodule:: biotite.structure
#
# Atom arrays and stacks can be indexed in a similar way a
# :class:`ndarray` is indexed.
# In fact, the index is propagated to the coordinates and the annotation
# arrays.
# Therefore, all *NumPy* compatible types of indices can be used,
# like boolean arrays, index arrays/lists, slices and, of course,
# integer values.
# Integer indices have a special role here, as they reduce the
# dimensionality of the data type:
# Indexing an :class:`AtomArrayStack` with an integer results in an
# :class:`AtomArray` at the specified frame, indexing an :class:`AtomArray`
# with an integer yields the specified :class:`Atom`.
# Iterating over arrays and stacks reduces the dimensionality in an
# analogous way.
# Let's demonstrate indexing with the help of the structure of *TC5b*.

from tempfile import gettempdir
import biotite.structure as struc
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio

file_path = rcsb.fetch("1l2y", "mmtf", gettempdir())
stack = strucio.load_structure(file_path)
print(type(stack).__name__)
print(stack.shape)
# Get the third model
array = stack[2]
print(type(array).__name__)
print(array.shape)
# Get the fifth atom
atom = array[4]
print(type(atom).__name__)
print(atom.shape)

########################################################################
# :func:`load_structure()` gives us an :class:`AtomArrayStack`.
# The first indexing step reduces the stack to an atom array and the
# second indexing step reduces the array to a single atom.
# The `shape` attribute gives the number of models and atoms, similarly
# to the `shape` attribute of :class:`ndarray` objects.
# Alternatively, the :func:`stack_depth()` or :func:`array_length()`
# methods can be used to get the number of models or atoms,
# respectively.
#
# The following code section shows some examples for how an atom array
# can be indexed.

# Get the first atom
atom = array[0]
# Get a subarray containing the first and third atom
subarray = array[[0,2]]
# Get a subarray containing a range of atoms using slices
subarray = array[100:200]
# Filter all carbon atoms in residue 1
subarray = array[(array.element == "C") & (array.res_id == 1)]
# Filter all atoms where the X-coordinate is smaller than 2
subarray = array[array.coord[:,0] < 2]

########################################################################
# An atom array stack can be indexed in a similar way, with the
# difference, that the index specifies the frame(s).

# Get an atom array from the first model
subarray = stack[0]
# Get a substack containing the first 10 models
substack = stack[:10]

########################################################################
# Stacks also have the speciality, that they can handle 2-dimensional
# indices, where the first dimension specifies the frame(s) and the
# second dimension specifies the atom(s).

# Get the first 100 atoms from the third model
subarray = stack[2, :100]
# Get the first 100 atoms from the models 3, 4 and 5
substack = stack[2:5, :100]
# Get the first atom in the second model
atom = stack[1,0]
# Get a stack containing arrays containing only the first atom
substack = stack[:, 0]

########################################################################
# Furthermore, :mod:`biotite.structure` contains advanced filters,
# that create boolean masks from an atom array using specific criteria.
# Here is a small example.

backbone = array[struc.filter_backbone(array)]
print(backbone.atom_name)

########################################################################
# If you would like to know which atoms are in proximity to specific
# coordinates, have a look at the :class:`CellList` class.
# 
# .. warning:: For annotation editing Since :class:`AnnotatedSequence` objects use base position
#    indices and :class:`Sequence` objects use array position indices,
#    you will get different results for ``annot_seq[n:m].sequence`` and
#    ``annot_seq.sequence[n:m]``.
#
# Representing bonds
# ------------------
# 
# Up to now we only looked into atom arrays whose atoms are merely
# described by its coordinates and annotations.
# But there is more: Chemical bonds can be described, too, using a
# :class:`BondList`!
# 
# Consider the following case: Your atom array contains four atoms:
# *N*, *CA*, *C* and *CB*. *CA* is a central atom that is connected to
# *N*, *C* and *CB*.
# A :class:`BondList` is created by passing a :class:`ndarray`
# containing pairs of integers, where each integer represents an index
# in a corresponding atom array.
# The pairs indicate which atoms share a bond.
# Additionally, it is required to specify the number of atoms in the
# atom array. 

from tempfile import gettempdir
import biotite.structure as struc

array = struc.array([
    struc.Atom([0,0,0], atom_name="N"),
    struc.Atom([0,0,0], atom_name="CA"),
    struc.Atom([0,0,0], atom_name="C"),
    struc.Atom([0,0,0], atom_name="CB")
])
print("Atoms:", array.atom_name)
bond_list = struc.BondList(
    array.array_length(),
    np.array([[1,0], [1,2], [1,3]])
)
print("Bonds (indices):")
print(bond_list.as_array())
print("Bonds (atoms names):")
print(array.atom_name[bond_list.as_array()[:, :2]])
ca_bonds, ca_bond_types = bond_list.get_bonds(1)
print("Bonds of CA:", array.atom_name[ca_bonds])

########################################################################
# When you look at the internal :class:`ndarray`
# (as given by :func:`BondList.as_array()`), you see a third column
# containing zeros.
# This column describes each bond with values from the :class:`BondType`
# enum: *0* corresponds to ``BondType.ANY``, which means that the type
# of the bond is undefined.
# This makes sense, since we did not define the bond types, when we
# created the bond list.
# The other thing that has changed is the index order:
# Each bond is sorted so that the index with the lower index is the
# first element.
#
# Although a :class:`BondList` uses a :class:`ndarray` under the hood,
# indexing works a little bit different:
# The indexing operation is not applied to the internal
# :class:`ndarray`, instead it behaves like the same indexing operation
# was applied to a corresponding atom array:
# The bond list adjusts its indices so that they still point to the same
# atoms as before.
# Bonds that involve at least one atom, that has been removed, are
# deleted as well.
# We will try that by deleting the *C* atom.

mask = (array.atom_name != "C")
sub_array = array[mask]
sub_bond_list = bond_list[mask]
print("Atoms:", sub_array.atom_name)
print("Bonds (indices):")
print(sub_bond_list.as_array())
print("Bonds (atoms names):")
print(sub_array.atom_name[sub_bond_list.as_array()[:, :2]])

########################################################################
# As you see, the the bonds involving the *C* (only a single one) are
# removed and the remaining indices are shifted.
# 
# Connecting atoms and bonds
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# We do not have to index the atom array and the bond list
# separately.
# For convenience reasons you can associate a :class:`BondList` to an
# :class:`AtomArray` via the ``bonds`` attribute.
# If no :class:`BondList` is associated, ``bonds`` is ``None``.
# Every time the atom array is indexed, the index is also applied to the
# associated bond list. 
# The same behavior applies to concatenations, by the way.

array.bonds = bond_list
sub_array = array[array.atom_name != "C"]
print("Bonds (atoms names):")
print(sub_array.atom_name[sub_array.bonds.as_array()[:, :2]])

########################################################################
# Note, that some functionalities in *Biotite* even require that the
# input structure has an associated :class:`BondList`. 
#
# Reading and writing bonds
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Up to now the bond information has been created manually, which is
# infeasible for larger molecules.
# Instead bond information can be loaded from and saved to some file
# formats, including MMTF and MOL files.
# We'll try that on the structure of *TC5b* and look at the bond
# information of the third residue, a tyrosine.

import biotite.database.rcsb as rcsb
import biotite.structure.io.mmtf as mmtf

file_path = rcsb.fetch("1l2y", "mmtf", gettempdir())
mmtf_file = mmtf.MMTFFile.read(file_path)
# Essential: set the 'include_bonds' parameter to true
stack = mmtf.get_structure(mmtf_file, include_bonds=True)
tyrosine = stack[:, (stack.res_id == 3)]
print("Bonds (indices):")
print(tyrosine.bonds)
print("Bonds (atoms names):")
print(tyrosine.atom_name[tyrosine.bonds.as_array()[:, :2]])

########################################################################
# Since we loaded the bond information from a MMTF file, the bond types
# are also defined:
# Here we have both, ``BondType.SINGLE`` and ``BondType.DOUBLE``
# bonds (*1* and *2*, respectively).
#
# Bond information can also be automatically inferred from an
# :class:`AtomArray` or :class:`AtomArrayStack`:
# :func:`connect_via_residue_names()` is able to connect atoms in
# all residues that appear in the
# `Chemical Component Dictionary <https://www.wwpdb.org/data/ccd>`_,
# comprising every molecule that appears in any PDB entry.
# In contrast, :func:`connect_via_distances()` uses only distances
# between atoms to infer the bonds.
# However, this function creates only ``BondType.ANY`` bonds and might
# connect wrong atoms in rare cases.
# Since both functions have the mentioned limitations it is usually
# better to read bond information directly from file, if available.

import biotite.structure.io.pdb as pdb

file_path = rcsb.fetch("1l2y", "pdb", gettempdir())
pdb_file = pdb.PDBFile.read(file_path)
stack = pdb_file.get_structure()
stack.bonds = struc.connect_via_residue_names(stack)
tyrosine = stack[:, (stack.res_id == 3)]
print("Bonds (indices):")
print(tyrosine.bonds)
print("Bonds (atoms names):")
print(tyrosine.atom_name[tyrosine.bonds.as_array()[:, :2]])


########################################################################
# Simulation boxes and unit cells
# -------------------------------
#
# Depending on the source of the macromolecular structure, there might
# be an associated unit cell or simulation box.
# In this package such boxes are represented by *(3,3)*-shaped
# :class:`ndarray` objects, where each element in the array is one of
# the three vectors spanning the box or unit cell.
# Let's create an orthorhombic box from the vector lengths and the
# angles between the vectors.

from tempfile import gettempdir
import numpy as np
import biotite.structure as struc

# The function uses angles in radians
box = struc.vectors_from_unitcell(10, 20, 30, np.pi/2, np.pi/2, np.pi/2)
print("Box:")
print(box)
print("Box volume:", struc.box_volume(box))
print("Is the box orthogonal?:", struc.is_orthogonal(box))
cell = struc.unitcell_from_vectors(box)
print("Cell:")
print(cell)

########################################################################
# An atom array can have an associated box, which is used in functions,
# that consider periodic boundary conditions.
# Atom array stacks require a *(m,3,3)*-shaped :class:`ndarray`, 
# that contains the box vectors for each model.
# The box is accessed via the `box` attribute, which is ``None`` by
# default.
# When loaded from a structure file, the box described in the file is
# automatically used.

import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio

array = struc.AtomArray(length=100)
print(array.box)
array.box = struc.vectors_from_unitcell(10, 20, 30, np.pi/2, np.pi/2, np.pi/2)
print(array.box)
file_path = rcsb.fetch("3o5r", "mmtf", gettempdir())
array = strucio.load_structure(file_path)
print(array.box)

########################################################################
# When loading a trajectory from an MD simulation, the molecules are
# often fragmented over the periodic boundary.
# While a lot of analysis functions can handle such periodic boundary
# conditions automatically, some require completed molecules.
# In this case you should use :func:`remove_pbc()`.

array = struc.remove_pbc(array)

########################################################################
# Structure manipulation
# ----------------------
#
# The most basic way to manipulate a structure is to edit the
# annotation arrays or coordinates directly.

from tempfile import gettempdir
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf

file_path = rcsb.fetch("1l2y", "mmtf", gettempdir())
mmtf_file = mmtf.MMTFFile.read(file_path)
structure = mmtf.get_structure(mmtf_file, model=1)
print("Before:")
print(structure[structure.res_id == 1])
print()
structure.coord += 100
print("After:")
print(structure[structure.res_id == 1])

########################################################################
# *Biotite* provides also some transformation functions, for example
# :func:`rotate()` for rotations about the *x*-, *y*- or *z*-axis.

structure = mmtf.get_structure(mmtf_file, model=1)
print("Before:")
print(structure[structure.res_id == 1])
print()
# Rotation about z-axis by 90 degrees
structure = struc.rotate(structure, [0, 0, np.deg2rad(90)])
print("After:")
print(structure[structure.res_id == 1])

########################################################################
# For a complete list of transformation functions have a look in the
# :doc:`API reference </apidoc/biotite.structure>`.
# 
# Structure analysis
# ------------------
# 
# This package would be almost useless, if there wasn't some means to
# analyze your structures.
# Therefore, *Biotite* offers a bunch of functions for this purpose,
# reaching from simple bond angle and length measurements to more
# complex characteristics, like accessible surface area and
# secondary structure.
# The following section will introduce you to some of these functions,
# which should be applied to that good old structure of *TC5b*.
# 
# The examples shown in this section are only a small glimpse into the
# *structure* analysis toolset.
# Have a look into the :doc:`API reference </apidoc/biotite.structure>`
# for more information.
# 
# Geometry measures
# ^^^^^^^^^^^^^^^^^
# 
# Let's start with measuring some simple geometric characteristics,
# for example atom distances of CA atoms.

from tempfile import gettempdir
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb

file_path = rcsb.fetch("1l2y", "mmtf", gettempdir())
stack = strucio.load_structure(file_path)
# Filter only CA atoms
stack = stack[:, stack.atom_name == "CA"]

# Calculate distance between first and second CA in first frame
array = stack[0]
print("Atom to atom:", struc.distance(array[0], array[1]))
# Calculate distance between the first atom
# and all other CA atoms in the array
print("Array to atom:")
array = stack[0]
print(struc.distance(array[0], array))
# Calculate pairwise distances between the CA atoms in the first frame
# and the CA atoms in the second frame
print("Array to array:")
print(struc.distance(stack[0], stack[1]))
# Calculate the distances between all CA atoms in the stack
# and the first CA atom in the first frame
# The resulting array is too large, therefore only the shape is printed
print("Stack to atom:")
print(struc.distance(stack, stack[0,0]).shape)
# And finally distances between two adjacent CA in the first frame
array = stack[0]
print("Adjacent CA distances")
print(struc.distance(array[:-1], array[1:]))

########################################################################
# Like some other functions in :mod:`biotite.structure`, we are able to
# pick any combination of an atom, atom array or stack. Alternatively
# :class:`ndarray` objects containing the coordinates can be provided.
# 
# Furthermore, we can measure bond angles and dihedral angles.

# Calculate angle between first 3 CA atoms in first frame
# (in radians)
print("Angle:", struc.angle(array[0],array[1],array[2]))
# Calculate dihedral angle between first 4 CA atoms in first frame
# (in radians)
print("Dihedral angle:", struc.dihedral(array[0],array[1],array[2],array[4]))

########################################################################
# .. note:: The :func:`distance()`, :func:`angle()` and
#    :func:`dihedral()` functions all have their :func:`pair_...()`
#    counterparts, that take an atom array (stack) and
#    pairs/triplets/quadruplets of atom indices, of which the
#    distance/angle should be calculated.
#    Both variants can be setup to consider periodic boundary conditions
#    by setting the `box` or `periodic` parameter, respectively.
# 
# In some cases one is interested in the dihedral angles of the peptide
# backbone, :math:`\phi`, :math:`\psi` and :math:`\omega`.
# In the following code snippet we measure these angles and create a
# simple Ramachandran plot for the first frame of *TC5b*.

import matplotlib.pyplot as plt
import numpy as np

array = strucio.load_structure(file_path)[0]
phi, psi, omega = struc.dihedral_backbone(array)
plt.plot(phi * 360/(2*np.pi), psi * 360/(2*np.pi),
        marker="o", linestyle="None")
plt.xlim(-180,180)
plt.ylim(-180,180)
plt.xlabel("$\phi$")
plt.ylabel("$\psi$")

########################################################################
# Comparing structures
# ^^^^^^^^^^^^^^^^^^^^
# 
# Now we want to calculate a measure of flexibility for each residue in
# *TC5b*. The *root mean square fluctuation* (RMSF) is a good value for
# that.
# It represents the deviation for each atom in all models relative
# to a reference model, which is usually the averaged structure.
# Since we are only interested in the backbone flexibility, we consider
# only CA atoms.
# Before we can calculate a reasonable RMSF, we have to superimpose each
# model on a reference model (we choose the first model),
# which minimizes the *root mean square deviation* (RMSD).

stack = strucio.load_structure(file_path)
# We consider only CA atoms
stack = stack[:, stack.atom_name == "CA"]
# Superimposing all models of the structure onto the first model
stack, transformation_tuple = struc.superimpose(stack[0], stack)
print("RMSD for each model to first model:")
print(struc.rmsd(stack[0], stack))
# Calculate the RMSF relative to the average of all models
rmsf = struc.rmsf(struc.average(stack), stack)
# Plotting stuff
plt.plot(np.arange(1,21), rmsf)
plt.xlim(0,21)
plt.xticks(np.arange(1,21))
plt.xlabel("Residue")
plt.ylabel("RMSF")

########################################################################
# As you can see, both terminal residues are most flexible.
# 
# Calculating accessible surface area
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Another interesting value for a protein structure is the
# *solvent accessible surface area* (SASA) that indicates whether an
# atom or residue is on the protein surface or buried inside the
# protein.
# The function :func:`sasa()` approximates the SASA for each
# atom.
# Then we sum up the values for each residue, to get the
# residue-wise SASA.
# 
# Besides other parameters, you can choose between different
# Van-der-Waals radii sets:
# *Prot0r*, the default set, is a set that defines radii for
# non-hydrogen atoms, but determines the radius of an atom based on the
# assumed amount of hydrogen atoms connected to it.
# Therefore, *ProtOr* is suitable for structures with missing hydrogen
# atoms, like crystal structures.
# Since the structure of *TC5b* was elucidated via NMR, we can assign a
# radius to every single atom (including hydrogens), hence we use the
# *Single* set.

array = strucio.load_structure(file_path)[0]
# The following line calculates the atom-wise SASA of the atom array
atom_sasa = struc.sasa(array, vdw_radii="Single")
# Sum up SASA for each residue in atom array
res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)
# Again plotting stuff
plt.plot(np.arange(1,21), res_sasa)
plt.xlim(0,21)
plt.xticks(np.arange(1,21))
plt.xlabel("Residue")
plt.ylabel("SASA")

########################################################################
# Secondary structure determination
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# *Biotite* can also be used to assign
# *secondary structure elements* (SSE) to a structure with the
# :func:`annotate_sse()` function.
# An ``'a'`` means alpha-helix, ``'b'`` beta-sheet, and ``'c'`` means
# coil.

array = strucio.load_structure(file_path)[0]
# Estimate secondary structure
sse = struc.annotate_sse(array, chain_id="A")
# Pretty print
print("".join(sse))