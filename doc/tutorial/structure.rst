Going 3D - The Structure subpackage
-----------------------------------
   
``structure`` is a Biopython subpackage for handling protein structures.
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
    
Based ony the implementation in `ndarrays`, this package furthermore
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
Additionally you can specify an arbitrary amount of annotations.
In most cases you won't work with `Atom` instances and in even fewer cases
`Atom` instances are created as it is done in the above example.

If you want to work with an entire molecular structure, containing an arbitrary
amount of atoms, you have to use so called atom arrays.
An atom array can be seen as an array of atom instances (hence the name).
But rather than storing `Atom` instances in a list or `ndarray`, an `AtomArray`
instance contains one `ndarray` for each annotation and the coordinates.
In order to see this in action, we first have to create an array from the atoms
we constructed before. The we can access the annotations and coordinates of the
atom array simply by specifying the attribute:

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
    
The `struc.array()` builder function takes any iterable object containg atom
instances, if you wanted to, you could even use another `AtomArray`.
An alternative way of constructing an array would be creating an
`AtomArray` by using its constructor, which fills the annotation arrays and
coordinates with the type respective *zero* value.
In our example all annotation arrays have a length of 3, since we used
3 atoms tp create it. A structure containing *n* atoms, is represented by
annotation arrays of length *n* and coordinates of shape *(n,3)*.

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
from PDB files, yet. The usage of PDBx/mmCIF files in favor of PDB files is
recommended, anyway. This format is a modern alternative to the PDB format
and will replace it someday. It solves limitations of the PDB format,
that arise from the column restrictions. Furthermore much more additional
information is stored in these files. In contrast to PDB files, Biopython
can read the entire content of PDBx/mmCIF files, which can accessed in a 
dictionary like manner.
At first we read the file similarily to before:

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

The first index contains data block and the category name. The data black could
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
`AtomArray`, you have to specifiy the `model` parameter.

For Biopython internal storage of structures *npz* files are recommended.
These are simply binary files, that are used by `numpy`. in case of atom arrays
and stacks, the annotation arrays and coordinates are written/read to/from
*npz* files via the `NpzFile` class. Since no expensive data conversion has
to be performed, this format is the fastest way to save and load atom arrays
and stacks.

Reading trajectory files
""""""""""""""""""""""""

If the package `MDtraj` is installed Biopython provides an read/write
interface for different trajectory file formats. More information can be found
in the API reference.

Array access and filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning:: Creating a subarray or substack by indexing, does not necessarily
   copy the coordinates and annotation arrays. If possible only *array views*
   are created. Look into the `numpy` documentation for furher details. If you
   want to ensure, that you are working with a copy, use the `copy()` method
   after indexing.

Structure analysis
^^^^^^^^^^^^^^^^^^

Geometry measures
"""""""""""""""""

Comparing structures
""""""""""""""""""""

Calculating accessible surface area
"""""""""""""""""""""""""""""""""""

Residue level operations

Secondary structure determination
"""""""""""""""""""""""""""""""""

Introduction
^^^^^^^^^^^^
Let's do some calculations on the miniprotein *TC5b* (PDB: 1L2Y). The structure
of this 20-residue protein has been elucidated via NMR, therefore the
corresponding PDB file consists of multiple (namely 38) models, each showing
another conformation.

At first, let's create the atom arrays and save it back into a PDB file.

.. code-block:: python

    import biopython.structure as struc
    import biopython.structure.io.pdb as pdb
    
    # Read the PDB file, which is in this case in the same folder
    pdb_file = pdb.PDBFile()
    pdb_file.read("1l2y.pdb")
    # Create the atom array stack from the models in the PDB file
    stack = pdb_file.get_structure()
    
    # Do some fancy stuff
    
    # Write stack back into the file
    pdb_file.set_structure(stack)
    pdb_file.write("1l2y.pdb")
    
    # Read file again and check, if both stacks have the same content
    pdb_file.read("1l2y.pdb")
    stack2 = pdb_file.get_structure()
    print("Are both stacks equal? " + str(stack == stack2))
	
Output:
	
``Are both stacks equal? True``

So far, so boring. This is a good place to mention that it is recommended to
use the modern PDBx/mmCIF format in favor of the PDB format. The parser
provides improved functionality, not only to read structures but also related
information. Therefore the PDBx parser will be used in the following examples.

Now let's do something more fancy. We calculate the RMSF of the
CA atoms of all models compared to the averaged structure.

.. code-block:: python

    import biopython.structure as struc
    import biopython.structure.io.pdbx as pdbx
    import numpy as np
    import matplotlib.pyplot as plt

    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read("1l2y.cif")
    stack = pdbx.get_structure(pdbx_file)
    # Filter the CA atoms
    stack = stack[:, stack.atom_name == "CA"]
    # Calculate the average structure
    average = struc.average(stack)
    # Calculate RMSF of all atom arrays in the stack compared to the average
    rmsf = struc.rmsf(average, stack)
    # Plot the results
    plt.plot(np.arange(1,21), rmsf)
    plt.xlim(0,20)
    plt.xticks(np.arange(1,21))
    plt.show()
	
Output:

.. image:: /static/assets/examples/rmsf.svg

As we can see, the CA position is most variable at the first and last position.
Now we test the superimposition capabilities: We extract the first and the
second model, then we move the second model and eventually superimpose the
moved second model on the first model. The first model is shown in red, the
second is shown in green before, and in orange after superimposition.
Additionally the RMSD between the first and the superimposed second model
is calculated.

.. code-block:: python

    import biopython.structure as struc
    import biopython.structure.io.pdbx as pdbx
    import numpy as np
    import matplotlib.pyplot as plt
    
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read("1l2y.cif")
    # Extract defined models
    array1 = pdbx.get_structure(pdbx_file, model=1)
    array2 = pdbx.get_structure(pdbx_file, model=2)
    # Translation and rotation of array2
    array2 = struc.translate(array2, (1,2,3))
    array2 = struc.rotate(array2, (1,2,3))
    # Superimpose array2 on array1
    fit_array2, transformation = struc.superimpose(array1, array2)
    # Calculate RMSD
    print("RMSD = " + str(struc.rmsd(array1, fit_array2)))
    # Visualize the structures' backbones
    fig = plt.figure()
    viewer = struc.simple_view(fig, [array1, array2, fit_array2])
    fig.tight_layout()
    plt.show()

Output:

``RMSD = 1.9548087935``

.. image:: /static/assets/examples/superimpose.svg

And finally we want to create a Ramachandran plot of the first model in the
structure.

.. code-block:: python

    import biopython.structure as struc
    import biopython.structure.io.pdbx as pdbx
    import numpy as np
    import matplotlib.pyplot as plt
    
    pdbx_file = pdbx.PDBxFile()
    pdbx_file.read("1l2y.cif")
    array = pdbx.get_structure(pdbx_file, model=1)
    # Calculate the backbone dihedral angles in chain "A" (only chain)
    psi, omega, phi = struc.dihedral_backbone(array, "A")
    # Plot the results
    plt.plot(phi * 360/(2*np.pi), psi * 360/(2*np.pi), linestyle="None", marker="o")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.show()
	
Output:
	
.. image:: /static/assets/examples/dihedral.svg
