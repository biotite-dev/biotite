"""
Biological assembly of a structure
==================================

Often the biological assembly (or biological unit) reveals the complete
picture of a protein function, may it be a viral capsid or a
microfilament.
However, the usual atom records in an *PDB* or *PDBx* file usually
describe only the asymmetric unit.
For large complexes the asymmetric unit may only display one monomer or
one small subcomplex.
Multiple copies of the asymmetric unit must be geometrically arranged to
build the assembly.

In order to get the entire assembly, the *PDBx* files provided by the
*RCSB PDB* (either in *CIF* or *BinaryCIF* format) contain the following
fields:

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
capsid from the lambda phage.

At first we will check, which assemblies are available to us.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import biotite.database.rcsb as rcsb
import biotite.interface.pymol as pymol_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

PDB_ID = "7VII"


pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(PDB_ID, "bcif"))

assemblies = pdbx.list_assemblies(pdbx_file)
print("ID    name")
print()
for assembly_id, name in assemblies.items():
    print(f"{assembly_id:2}    {name}")

########################################################################
# ``'complete icosahedral assembly'`` sounds good.
# In fact, often the first assembly is the complete one.
# Hence, the :func:`get_assembly()` function builds the first assembly
# by default.
# Since we know the ID we want (``'1'``), we will provide it to this
# function anyway.
# It returns the chosen assembly as :class:`AtomArray`.
# Note that the assembly ID is a string, not an integer.

assembly = pdbx.get_assembly(
    pdbx_file,
    assembly_id="1",
    model=1,
    # To identify later which atoms belong to which protein type
    extra_fields=["label_entity_id"],
)

print("Number of protein chains:", struc.get_chain_count(assembly))

########################################################################
# The assembly consists of two different protein types, so called entities.
# Each entity may be represented by multiple chain IDs.

entity_info = pdbx_file.block["entity"]
print("ID    description")
print()
for entity_id, description in zip(
    entity_info["id"].as_array(), entity_info["pdbx_description"].as_array()
):
    print(f"{entity_id:2}    {description}")

########################################################################
# Now we could do some analysis on the biological unit.
# But for this example we will visualize the entire assembly.

# Show capsid structure as CA spheres to increase rendering speed
assembly = assembly[assembly.atom_name == "CA"]
pymol_object = pymol_interface.PyMOLObject.from_structure(assembly)
pymol_object.color("biotite_dimgreen", assembly.label_entity_id == "1")
pymol_object.color("biotite_lightorange", assembly.label_entity_id == "2")
pymol_object.set("sphere_scale", 2.0)
pymol_object.show_as("spheres")
pymol_object.zoom(buffer=25)
pymol_interface.show((1500, 1500))
