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


def transform(coord, struct_oper_list, index):
    rotation_matrix = np.array([
        [float(struct_oper_list[f"matrix[{i}][{j}]"][index]) for j in (1,2,3)]
        for i in (1,2,3)
    ])
    translation_vector = np.array([
        float(struct_oper_list[f"vector[{i}]"][index]) for i in (1,2,3)
    ])
    # Rotate
    coord = np.matmul(rotation_matrix, coord.T).T
    # Translate
    coord += translation_vector
    return coord


def repeat(template, coord):
    orig_length = template.array_length()
    new_length = coord.shape[-2]
    if new_length % orig_length != 0:
        raise ValueError(
            "The length of the new coordinates must be a multiple of the "
            "template length"
        )
    repetitions = new_length // orig_length

    repeated = struc.AtomArray(new_length)
    
    for category in template.get_annotation_categories():
        annot = np.tile(template.get_annotation(category), repetitions)
        repeated.set_annotation(category, annot)
    if template.bonds is not None:
        bonds = template.bonds
        for _ in range(repetitions-1):
            bonds += template.bonds
        repeated.bonds = bonds
    if template.box is not None:
        repeated.box = template.box.copy()
    repeated.coord = coord
    
    return repeated




#pdbx_file = pdbx.PDBxFile()
#pdbx_file.read(rcsb.fetch("4uft", "mmcif"))
###
import biotite
file_name = rcsb.fetch("1m4x", "mmcif", ".")
pdbx_file = pdbx.PDBxFile()
pdbx_file.read(file_name)
###

structure = pdbx.get_structure(pdbx_file, model=1)
coord = structure.coord


struct_oper_list = pdbx_file["pdbx_struct_oper_list"]

biological_unit_coord = []
# For each symmetric capsid component, build the asymmetric subunit
for i, transformation_type_i in enumerate(struct_oper_list["type"]):
    if transformation_type_i == "point symmetry operation":
        coord_sym = transform(coord, struct_oper_list, i)
        for j, transformation_type_j in enumerate(struct_oper_list["type"]):
            if transformation_type_j == "build point asymmetric unit":
                coord_asym = transform(coord_sym, struct_oper_list, j)
                biological_unit_coord.append(coord_asym)


biological_unit_coord = np.concatenate(biological_unit_coord, axis=0)
biological_unit = repeat(structure, biological_unit_coord)


strucio.save_structure("unit.pdb", biological_unit)