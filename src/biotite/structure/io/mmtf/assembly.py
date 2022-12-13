# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mmtf"
__author__ = "Patrick Kunzmann"
__all__ = ["get_assembly"]


import numpy as np
from .convertfile import get_structure
from ...chains import get_chain_starts
from ...util import matrix_rotate
from ....file import InvalidFileError


def list_assemblies(file):
    return [assembly["name"] for assembly in file["bioAssemblyList"]]


def get_assembly(file, assembly_id=None, model=None, altloc="first",
                 extra_fields=[], include_bonds=False):
    structure = get_structure(
        file, model, altloc, extra_fields, include_bonds
    )

    # Get transformations for chosen assembly
    selected_assembly = None
    if not "bioAssemblyList" in file:
        raise InvalidFileError(
                "File does not contain assembly information "
                "(missing 'bioAssemblyList')"
            )
    for assembly in file["bioAssemblyList"]:
        current_assembly_id = assembly["name"]
        transform_list = assembly["transformList"]
        if assembly_id is None or current_assembly_id == assembly_id:
            selected_assembly = transform_list
            break
    if selected_assembly is None:
        raise KeyError(
            f"The assembly ID '{assembly_id}' is not found"
        )
    
    # In most cases the transformations in an assembly applies to all
    # atoms equally ('apply_to_all == True')
    # If this is the case, the selection of atoms for each
    # transformation can be omitted, improving the performance
    chain_index_count = len(file["chainNameList"])
    apply_to_all = True
    for transformation in selected_assembly:
        # If the number of affected chains matches the number of total
        # chains, all atoms are affected
        if len(transformation["chainIndexList"]) != chain_index_count:
            apply_to_all = False
    # If the transformations in the assembly do not apply to all atoms,
    # but only to certain chains we need the ranges of these chains
    # in the base structure (the asymmetric unit)
    if not apply_to_all:
        chains_starts = get_chain_starts(
            structure, add_exclusive_stop=True
        )
        # Furthermore the number of chains determined by Biotite via
        # 'get_chain_starts()' must corresponds to the number of chains
        # in the MMTF file
        # If this is not the case the assembly cannot be read using
        # this function due to the shortcoming in 'get_structure()'
        if len(chains_starts) != chain_index_count:
            raise NotImplementedError(
                "The structure file is not suitable for this function, as the "
                "number of chains in the file do not match the automatically "
                "detected number of chains"
            )
    
    # Apply transformations for set of chains (or all chains) and add
    # the transformed atoms to assembly
    assembly = None
    for transformation in selected_assembly:
        if apply_to_all:
            affected_coord = structure.coord
        else:
            # Mask atoms affected by this transformation
            affected_mask = np.zeros(structure.array_length(), dtype=bool)
            for chain_i in transformation["chainIndexList"]:
                chain_start = chains_starts[chain_i]
                chain_stop = chains_starts[chain_i+1]
                affected_mask[chain_start : chain_stop] = True
            affected_coord = structure.coord[..., affected_mask, :]
        # Apply the transformation
        transformed_coord = _apply_transformation(
            affected_coord, transformation["matrix"]
        )
        sub_assembly = structure.copy()
        sub_assembly.coord = transformed_coord
        # Add transformed coordinates to assembly
        if assembly is None:
            assembly = sub_assembly
        else:
            assembly += sub_assembly
    
    return assembly


def _apply_transformation(coord, mmtf_matrix):
    # Obtain matrix from flattened form
    matrix = np.array(mmtf_matrix).reshape(4, 4)
    # Separate rotation and translation part
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    coord = matrix_rotate(coord, rotation)
    coord += translation
    return coord