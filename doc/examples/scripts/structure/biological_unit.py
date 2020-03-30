"""
Biological assembly of a structure
==================================

Often the biological assembly (or biological unit) reveals the complete
picture of a protein function, may it be a viral capsid or a
microfilament.
However, the usual records in an *PDB*/*mmCIF*/*MMTF* usually describe
only the asymmetric unit.
For large complexes the asymmetric unit may only display one monomer or
one small subcomplex.
Multiple copies of the asymmetric unit must be geometrically arranged to
build the assembly.

In order to get the entire assembly, the *mmCIF* files provided by the
*RCSB PDB* contain the following fields:

    - ``pdbx_struct_assembly`` - General information about the
      assemblies
    - ``pdbx_struct_assembly_prop`` - More specific properties
    - ``pdbx_struct_oper_list`` - A list of tranformation operations
    - ``pdbx_struct_assembly_gen`` - Which tranformation operations from
      ``pdbx_struct_oper_list`` need to be applied for the assemblies

More information about biological assemblies is provided by the
*RCSB PDB* on
`this page <https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/biological-assemblies>`_.

In this example, we will create the complete biological assembly of the
capsid from *Paramecium bursaria Chlorella virus type 1*
- a homo-5040-mer!

At first we well check, which assemblies are available to us.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb


pdbx_file = pdbx.PDBxFile()
pdbx_file.read(rcsb.fetch("1M4X", "mmcif"))

assemblies = pdbx.get_assembly_list(pdbx_file)
print("ID    name")
print()
for assembly_id, name in assemblies.items():
    print(f"{assembly_id:2}    {name}")

########################################################################
# `'complete icosahedral assembly'` sounds good.
# In fact, often the first assembly is the complete assembly.
# Hence, the :func:`get_assembly()` function, that we will use the get
# the chosen assembly as :class:`AtomArray`, builds the first assembly
# by default.
# Since we know the assembly ID we want (``'1'``), we will provide it
# anyway.
# Note that the assembly ID is a string, not an integer.

biological_unit = pdbx.get_assembly(pdbx_file, assembly_id="1", model=1)
print("Number of protein chains:", struc.get_chain_count(biological_unit))

########################################################################
# Now we could do some analysis on the assembly.
# But for this example we will simply save the entire assembly as *PDB*
# file for later visualization.

# For brevity, save only CA atoms to file for visualization with PyMOL
#biological_unit = biological_unit[biological_unit.atom_name == "CA"]
#strucio.save_structure("biological_unit.pdb", biological_unit)
# biotite_static_image = biological_unit.png