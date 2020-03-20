"""
Biological unit of a structure
==============================

Quaternary structure
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb


pdbx_file = pdbx.PDBxFile()
pdbx_file.read(rcsb.fetch("4uft", "mmcif"))

structure = pdbx.get_structure(pdbx_file, model=1)

transformations = pdbx_file["pdbx_struct_oper_list"]
biological_unit = struc.AtomArray(0)
for k in range(len(transformations["id"])):
    if transformations["type"][k] == "helical symmetry operation":
        rotation_matrix = np.array([
            [float(transformations[f"matrix[{i}][{j}]"][k]) for j in (1,2,3)]
            for i in (1,2,3)
        ])
        translation_vector = np.array([
            float(transformations[f"vector[{i}]"][k]) for i in (1,2,3)
        ])
        monomer = structure.copy()
        # Rotate
        monomer.coord = np.matmul(rotation_matrix, monomer.coord.T).T
        # Translate
        monomer.coord += translation_vector
        biological_unit += monomer

strucio.save_structure("unit.pdb", biological_unit)