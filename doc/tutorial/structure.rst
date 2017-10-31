Going 3D - The Structure subpackage
-----------------------------------
   
``structure`` is a *Biopython* subpackage for handling molecular structures.
This subpackage enables efficient and easy handling of protein structure data
by representing atom attributes in `numpy` `ndarrays`. These atom attributes
include so called *annotations* (polypetide chain id, residue id, residue name,
hetero residue information, atom name, element, etc.) and the atom coordinates.

The package contains mainly three types: `Atom`, `AtomArray` and
`AtomArrayStack`. An `Atom` contains data for a single atom, an `AtomArray`
stores data for an entire model and `AtomArrayStack` stores data for multiple
models, where each model contains the same atoms but differs in the atom
coordinates. Both, `AtomArrray` and `AtomArrayStack`, store the attributes
in `numpy` arrays. This approach has multiple advantages:
    
    - Convenient selection of atoms in a structure
      by using `numpy` style indexing
    - Fast calculations on structures using C-accelerated `ndarray` operations
    - Simple implementation of custom calculations
    
Based on the implementation in `ndarrays`, this package also
contains functions for structure analysis, manipulation and visualisation.

Creating structures
^^^^^^^^^^^^^^^^^^^

Let's begin by constructing some atoms:

.. code-block:: python

   import biopython.structure as struc
   atom1 = struc.Atom([0,0,0], chain_id="A", res_id=1, res_name="GLY",
                      hetero=False, atom_name="N", element="N")
   atom2 = struc.Atom([0,1,1], chain_id="A", res_id=1, res_name="GLY",
                      hetero=False, atom_name="CA", element="C")
   atom3 = struc.Atom([0,0,2], chain_id="A", res_id=1, res_name="GLY",
                       hetero=False, atom_name="C", element="C")

The first parameter are the coordinates (internally converted into an
`ndarray`), the other parameters are annotations.
The annotations shown in this example are mandatory: If you miss one of these,
Python will not complain, but some operations might not work properly
(especially true, when we go to atom arrays and stacks). The mandatory
annotation categories are originated in *ATOM* records in the PDB format.
Additionally, you can specify an arbitrary amount of annotations.
In most cases you won't work with `Atom` instances and in even fewer cases
`Atom` instances are created as it is done in the above example.

If you want to work with an entire molecular structure, containing an arbitrary
amount of atoms, you have to use so called atom arrays.
An atom array can be seen as an array of atom instances (hence the name).
But rather than storing `Atom` instances in a list or `ndarray`, an `AtomArray`
instance contains one `ndarray` for each annotation and the coordinates.
In order to see this in action, we first have to create an array from the atoms
we constructed before. Then we can access the annotations and coordinates of
the atom array simply by specifying the attribute:

.. code-block:: python

   array = struc.array([atom1, atom2, atom3])
   print(array.chain_id)
   print(array.res_id)
   print(array.atom_name)
   print(array.coord)
   print()
   print(array)

Output:

.. code-block:: none

   Chain ID: ['A' 'A' 'A']
   Residue ID: [1 1 1]
   Atom name: ['N' 'CA' 'C']
   [[ 0.  0.  0.]
    [ 0.  1.  1.]
    [ 0.  0.  2.]]
   
   A  1  GLY   False N  N  [ 0.  0.  0.]
   A  1  GLY   False CA C  [ 0.  1.  1.]
   A  1  GLY   False C  C  [ 0.  0.  2.]
    
The `struc.array()` builder function takes any iterable object containing atom
instances, if you wanted to, you could even use another `AtomArray`.
An alternative way of constructing an array would be creating an
`AtomArray` by using its constructor, which fills the annotation arrays and
coordinates with the type respective *zero* value.
In our example all annotation arrays have a length of 3, since we used
3 atoms to create it. A structure containing *n* atoms, is represented by
annotation arrays of length *n* and coordinates of shape *(n,3)*.

If you want to add further annotation categories to an array, at first you have
to call the `add_annotation()` or `set_annotation()` method. After that you can
access the new annotation array like any other annotation array.

In some cases, you might need to handle structures, where each atom is present
in multiple locations (multiple models in NMR structures, MD trajectories).
For the cases atom array stacks are used, which represent a list of atom
arrays. Since the atoms are the same for each frame, but only the coordinates
change, the annotation arrays in stacks are still the same length *n*
`ndarrays` as in atom arrays. However, a stack stores the coordinates in a
*(m,n,3)*-shaped `ndarray`, where *m* is the number of frames.
A stack is constructed analogous to the code snipped above. It is crucial
that all arrays, that should be stacked, have the same annotation arrays,
otherwise an exception is raised. For simplicity reasons, we create a
stack containing two identical frames, derived from the previous example.

.. code-block:: python

   stack = struc.stack([array, array.copy()])
   print(stack)

Output:

.. code-block:: none
   
   
   Model 1
   A  1  GLY   False N  N  [ 0.  0.  0.]
   A  1  GLY   False CA C  [ 0.  1.  1.]
   A  1  GLY   False C  C  [ 0.  0.  2.]
   
   
   Model 2
   A  1  GLY   False N  N  [ 0.  0.  0.]
   A  1  GLY   False CA C  [ 0.  1.  1.]
   A  1  GLY   False C  C  [ 0.  0.  2.]

Loading structures from file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually structures are not built from scratch using `Biopython`, but they are
read from a file. Probably the most popular one is the *PDB* format. For our
purpose, we will work on a protein structure as small as possible, namely
the miniprotein *TC5b* (PDB: 1L2Y). The structure of this 20-residue protein
(304 atoms) has been elucidated via NMR, therefore the corresponding PDB file
consists of multiple (namely 38) models, each showing another conformation.

At first we load the structure from a PDB file:

.. code-block:: python
   
   import biopython.structure.io.pdb as pdb
   file = pdb.PDBFile()
   file.read("path/to/1l2y.pdb")
   tc5b = file.get_structure()
   print(type(tc5b).__name__)
   print(tc5b.stack_depth())
   print(tc5b.array_length())

Output:

.. code-block:: none
   
   AtomArrayStack
   38
   304

The method `get_structure()` returns a stack only when multiple models exist
in the PDB file, otherwise the method returns an array. The following example
shows how to write an array or stack back into a PDB file:

.. code-block:: python
   
   file = pdb.PDBFile()
   file.set_structure(tc5b)
   file.write("path/to/1l2y_modified.pdb")

Other information (authors, secondary structure, etc.) cannot be extracted
from PDB files, yet. This is a good place to mention that it is recommended to
use the modern PDBx/mmCIF format in favor of the PDB format. It solves
limitations of the PDB format, that arise from the column restrictions.
Furthermore, much more additional information is stored in these files.
In contrast to PDB files, *Biopython* can read the entire content of PDBx/mmCIF
files, which can be accessed in a dictionary like manner.
At first, we read the file similarily to before:

.. code-block:: python
   
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")

Now we can access the data like a dictionary of dictionaries.

.. code-block:: python
   
   print(file["1L2Y", "audit_author"]["name"])

Output:

.. code-block:: none
   
   ['Neidigh, J.W.' 'Fesinmeyer, R.M.' 'Andersen, N.H.']

The first index contains the data block and the category name. The data block could
be omitted, since there is only one block in the file. This returns a
dictionary. If the category is in a *loop*, the dictionary contains `ndarrays`
of strings as values, otherwise the dictionary contains strings directly.
The second index specifies the name of the subcategory, which is used as key in
this dictionary and returns the corresponding `ndarray`.
Setting/adding a category in the file is done in a similar way:

.. code-block:: python
   
   file["audit_author"] = {"name" : ["Doe, Jane", "Doe, John"],
                           "pdbx_ordinal" : ["1","2"]}

In most applications only the structure itself (stored in the *atom_site*
category) is relevant. There are convenience functions that are used to
convert the *atom_site* category into an atom array/stack and vice versa.

.. code-block:: python
   
   tc5b = pdbx.get_structure(file)
   # Do some fancy stuff
   pdbx.set_structure(file, tc5b)

`get_structure()` creates automatically an `AtomArrayStack`, even if the file
actually contains only a single model. If you would like to have an
`AtomArray` instead, you have to specifiy the `model` parameter.

For *Biopython* internal storage of structures *npz* files are recommended.
These are simply binary files, that are used by `numpy`. In case of atom arrays
and stacks, the annotation arrays and coordinates are written/read to/from
*npz* files via the `NpzFile` class. Since no expensive data conversion has
to be performed, this format is the fastest way to save and load atom arrays
and stacks.

Since programmers are usually lazy and do not want to write more code than
necessary, there is a convenient function that unifies the forementioned
file formats. `get_structure_from()` takes a file path and outputs an array
(or stack, if the files contains multiple models). Internally, this function
uses the appropriate `File` class, depending on the file format.

.. code-block:: python
   
   import biopython.structure.io as strucio
   stack_from_cif = strucio.get_structure_from("path/to/1l2y.cif")
   stack_from_pdb = strucio.get_structure_from("path/to/1l2y.pdb")
   print("Are both stacks equal?", stack_from_cif == stack_from_pdb)

Output:

.. code-block:: none
   
   Are both stacks equal? True

Reading trajectory files
""""""""""""""""""""""""

If the package `MDtraj` is installed *Biopython* provides a read/write
interface for different trajectory file formats. More information can be found
in the API reference.

Array indexing and filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Atom arrays and stacks can be indexed in a similar way an `ndarray` is indexed.
In fact, the index is propagated to the coordinates and the annotation arrays.
Therefore, different kinds of indices can be used, like boolean arrays, lists
containing indices, slices and, of course, integer values. Integer indices have
a special role here, as they reduce the dimensionality of the data type:
Indexing an `AtomArrayStack` with an integer results in an `AtomArray` at the
specified frame, indexing an `AtomArray` with an integer yields the specified
`Atom`. Iterating over arrays and stacks reduces the dimensionality in an
analogous way.
Let's demonstrate indexing with the help of the structure of *TC5b*.

.. code-block:: python
   
   import biopython.structure as struc
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")
   stack = pdbx.get_structure(file)
   print(type(stack).__name__)
   array = stack[2]
   print(type(array).__name__)
   print(array.array_length())

Output:

.. code-block:: none
   
   AtomArrayStack
   AtomArray
   304
   

This `get_structure()` gives us an `AtomArrayStack`. Via the integer index,
we get the `AtomArray` representing the third model. The `array_length()`
method gives us the number of atoms in arrays and stacks and is equivalent
to the length of an atom array.
The following code section shows some examples for how an atom array can be
indexed.

.. code-block:: python
   
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

An atom array stack can be indexed in a similar way, with the difference, that
the index specifies the frame(s).

.. code-block:: python
   
   # Get an atom array from the first model
   subarray = stack[0]
   # Get a substack containing the first 10 models
   substack = stack[:10]

Stacks also have the speciality, that they can handle 2-dimensional indices,
where the first dimension specifies the frame and the second dimension
specifies the atom.

.. code-block:: python
   
   # Get the first 100 atoms from the third model
   subarray = stack[2, :100]
   # Get the first 100 atoms from the models 3, 4 and 5
   substack = stack[2:5, :100]
   # Get the first atom in the second model
   atom = stack[1,0]
   # Get a stack containing arrays containing only the first atom
   substack = stack[:, 0]

Furthermore, the package contains advanced filters, that create boolean masks
from an array using specific criteria. Here is a small example

.. code-block:: python
   
   backbone = array[struc.filter_backbone(array)]
   print(backbone.atom_name)

Output:

.. code-block:: none
   
   ['N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA'
    'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N'
    'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C'
    'N' 'CA' 'C' 'N' 'CA' 'C' 'N' 'CA' 'C']

If you would like to know which atoms are in proximity to specific coordinates,
have a look at the `AdjacencyMap` class.

.. warning:: Creating a subarray or substack by indexing, does not necessarily
   copy the coordinates and annotation arrays. If possible, only *array views*
   are created. Look into the `numpy` documentation for furher details. If you
   want to ensure, that you are working with a copy, use the `copy()` method
   after indexing.

Structure analysis
^^^^^^^^^^^^^^^^^^

This package would be almost useless, if there wasn't some means to analyze
your structures. Therefore, *Biopython* offers a bunch of functions for this
purpose, reaching from simple bond angle and length measurements to more
complex characteristics, like accessible surface area and secondary structure.
The following section will introduce you to some of these functions, which
should be applied to that good old structure of *TC5b*.

The examples shown in this section do not represent the full spectrum of
analysis tools in this package. Look into the API reference for more
information.

Geometry measures
"""""""""""""""""

Let's start with measuring some simple geometric characteristics, for example
atom distances of CA atoms:

.. code-block:: python
   
   import biopython.structure as struc
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")
   stack = pdbx.get_structure(file)
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

Output:

.. code-block:: none
   
   Atom to atom: 3.87639910226
   Array to atom:
   [  0.           3.8763991    5.57665975   5.03889055   6.31640919
      8.76681499   9.90813499  10.61481667  12.89033149  14.80667937
     13.50116443  16.87541054  18.72356614  17.22428861  19.11193308
     16.19300176  15.51475678  12.37781309  10.44593404  12.0589665 ]
   Array to array:
   [ 3.43441989  0.37241509  0.22178593  0.10823123  0.15207235  0.1701705
     0.22572771  0.47650498  0.2949322   0.1548354   0.28323488  0.40683903
     0.13555073  0.36768737  0.46464395  0.57544244  0.33707418  0.25703307
     0.34762192  0.38818681]
   Stack to atom:
   (38, 20)
   Adjacent CA distances
   [ 3.8763991   3.86050178  3.87147026  3.84557993  3.86660471  3.85851811
     3.88180293  3.86098705  3.89091814  3.86355497  3.88626993  3.87561298
     3.87466863  3.86554912  3.86627728  3.87766244  3.86038275  3.85824688
     3.86421907]

Like some other functions in this package, we are able to pick any combination
of an atom, atom array or stack. Alternatively `ndarrays` containing the
coordinates can be provided.

Furthermore, we can measure bond angles and dihedral angles:

.. code-block:: python
   
   # Calculate angle between first 3 CA atoms in first frame
   # (in radians)
   print("Angle:", struc.angle(array[0],array[1],array[2]))
   # Calculate dihedral angle between first 4 CA atoms in first frame
   # (in radians)
   print("Dihedral angle:", struc.dihedral(array[0],array[1],array[2],array[4]))

Output:

.. code-block:: none
   
   Angle: 1.60987082193
   Dihedral angle: 1.49037920852

In some cases one is interested in the dihedral angles of the peptide backbone,
*phi*, *psi* and *omega*. In the following code snippet we measure these angles
and create a Ramachandran plot for the first frame of *TC5b*.

.. code-block:: python
   
   import matplotlib.pyplot as plt
   import numpy as np
   import biopython.structure as struc
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")
   array = pdbx.get_structure(file, model=1)
   phi, psi, omega = struc.dihedral_backbone(array, chain_id="A")
   plt.plot(phi * 360/(2*np.pi), psi * 360/(2*np.pi),
            marker="o", linestyle="None")
   plt.xlim(-180,180)
   plt.ylim(-180,180)
   plt.xlabel("phi")
   plt.ylabel("psi")
   plt.show()

Output:

.. image:: /static/assets/figures/dihedral.svg

Comparing structures
""""""""""""""""""""

Now we want to calculate a measure of flexibility for each residue in *TC5b*.
The *root mean square fluctuation* (RMSF) is a good value for that. It
represents the deviation for each atom in all models relative to a reference
model, which is usually the averaged structure. Since we are only interested in
the backbone flexibility, we consider only CA atoms.
Before we can calculate a reasonable RMSF, we have to superimpose each model on
a reference model (we choose the first model), which minimizes the
*root mean square deviation* (RMSD).

.. code-block:: python
   
   import matplotlib.pyplot as plt
   import numpy as np
   import biopython.structure as struc
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")
   stack = pdbx.get_structure(file)
   # We consider only CA atoms
   stack = stack[:, stack.atom_name == "CA"]
   # Superimposing all models of the structure onto the first model
   stack, transformation_tuple = struc.superimpose(stack[0], stack)
   print("RMSD for each model to first model:")
   print(struc.rmsd(stack[0], stack))
   # Calculate the RMSF relative to average of all models
   rmsf = struc.rmsf(struc.average(stack), stack)
   # Plotting stuff
   plt.plot(np.arange(1,21), rmsf)
   plt.xlim(0,20)
   plt.xticks(np.arange(1,21))
   plt.xlabel("Residue")
   plt.ylabel("RMSF")
   plt.show()

Output:

.. code-block:: none
   
   RMSD for each model to first model:
   [ 0.          0.78426444  1.00757683  0.55180272  0.80663454  1.06066791
     0.87383705  0.62606424  1.00576561  0.81440804  0.87628298  1.35385898
     0.93278001  0.87600934  0.99357313  0.40626578  0.31801941  1.18389047
     1.23477073  0.89114465  0.55536526  0.73639392  0.78567399  1.10192568
     0.67228881  1.1605639   0.98213955  1.22808849  0.79269641  0.86854739
     0.93866682  0.83565702  0.61650375  0.97335428  1.03223981  0.55556655
     1.15175216  0.85585345]

.. image:: /static/assets/figures/rmsf.svg

As you can see, both terminal residues are most flexible.

Calculating accessible surface area
"""""""""""""""""""""""""""""""""""

Another interesting value for a protein structure is the
*solvent accessible surface area* (SASA) that indicates whether an atom or
residue is on the protein surface or buried inside the protein. The function
`sasa()` numerically calculates the SASA for each atom. Then we sum up the
values for each residue, to get the residue-wise SASA.

Besides other parameters, you can choose between different Van-der-Waals radii
sets:
*Prot0r*, the default set, is a set that defines radii for non-hydrogen atoms,
but determines the radius of an atom based on the assumed amount of hydrogen
atoms connected to it. Therefore, *Prot0r* is suitable for structures with
missing hydrogen atoms, like crystal structures.
Since the structure of *TC5b* was elucidated via NMR, we can assign a radius to
every single atom (including hydrogens), hence we use the *Single* set.

.. code-block:: python
   
   import matplotlib.pyplot as plt
   import numpy as np
   import biopython.structure as struc
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")
   array = pdbx.get_structure(file, model=1)
   # The following line calculates the atom-wise SASA of the atom array
   atom_sasa = struc.sasa(array, vdw_radii="Single")
   # Sum up SASA for each residue in atom array
   res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)
   # Again plotting stuff
   plt.plot(np.arange(1,21), res_sasa)
   plt.xlim(0,20)
   plt.xticks(np.arange(1,21))
   plt.xlabel("Residue")
   plt.ylabel("SASA")
   plt.show()

Output:

.. image:: /static/assets/figures/sasa.svg

Secondary structure determination
"""""""""""""""""""""""""""""""""

Biopython can also be used to assign *secondary structure elements* (SSE) to
a structure:

.. code-block:: python
   
   import biopython.structure as struc
   import biopython.structure.io.pdbx as pdbx
   file = pdbx.PDBxFile()
   file.read("path/to/1l2y.cif")
   array = pdbx.get_structure(file, model=1)
   # Estimate secondary structure
   sse = struc.annotate_sse(array, chain_id="A")
   # Pretty print
   print("".join(sse))

Output:

.. code-block:: none
   
   caaaaaaaaccccccccccc

An 'a' means alpha-helix, 'b' beta-sheet, and 'c' gamma-... just kidding,
'c' means coil.

