Going 3D - The Structure subpackage
-----------------------------------
   
``structure`` is a Biopython subpackage for handling protein structures.
This subpackage enables efficient and easy handling of protein structure data
by representation of atom attributes in `numpy` arrays. These atom attributes
include polypetide chain id, residue id, residue name, hetero residue
information, atom name (the *annotations*) and atom coordinates.

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
    
Based ony the implementation in `numpy` arrays, this package furthermore
contains functions for structure analysis, manipulation and visualisation.

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
