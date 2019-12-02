r"""
Ab initio assembly of a linear peptide
======================================
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import itertools
import numpy as np
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.info as info


N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.54
C_N_LENGTH  = 1.34


def assemble_peptide(sequence):
    backbone = struc.AtomArray(length=3*len(sequence))
    res_id = 0
    for i, bond_length, angle, atom_name in zip(
        range(backbone.array_length()),
        itertools.cycle([C_N_LENGTH, N_CA_LENGTH, CA_C_LENGTH]),
        itertools.cycle([120, -120]),
        itertools.cycle(["N", "CA", "C"])
    ):
        if atom_name == "N":
           res_id += 1 
        
        backbone.res_id[i] = res_id
        backbone.atom_name[i] = atom_name
        backbone.element[i] = atom_name[0]
        backbone.res_name[i] = "GLY"

        if i == 0:
            backbone.coord[i] = [0,0,0]
        elif i == 1:
            backbone.coord[i] = [bond_length,0,0]
        else:
            rot_axis = [0, 0, 1]
            backbone.coord[i] = struc.rotate_about_axis(
                atoms = backbone.coord[i-2],
                axis = rot_axis,
                angle = np.deg2rad(angle),
                support = backbone.coord[i-1]
            )
            # Scale vector to correct bond length
            bond_vector = backbone.coord[i] - backbone.coord[i-1]
            backbone.coord[i] = backbone.coord[i-1] \
                + bond_vector * bond_length / np.linalg.norm(bond_vector)


    backbone_extra = struc.AtomArray(length=0)
    for i, (residue, angle) in enumerate(zip(
        struc.residue_iter(backbone),
        itertools.cycle([120, -120]),
    )):
        rot_axis = [0, 0, 1]
        coord = struc.rotate_about_axis(
            atoms = residue.coord[-2],
            axis = rot_axis,
            angle = np.deg2rad(angle),
            support = residue.coord[-1]
        )
        backbone_extra += residue + struc.array([
            struc.Atom(
                coord, atom_name="O", element="O", res_id=i+1, res_name = "GLY"
            )
        ])
    backbone = backbone_extra
    print(backbone)


    full_atom_structure = struc.AtomArray(length=0)
    res_names = [seq.ProteinSequence.convert_letter_1to3(r) for r in sequence]
    for i, res_name in enumerate(res_names):
        residue = info.residue(res_name)
        
        backbone_for_res = backbone[backbone.res_id == i+1].copy()
        side_chain = residue[~np.isin(
            residue.atom_name,
            ["N", "CA", "C", "O", "OXT", "H", "H2", "H3", "HXT"]
        )]
        
        if i != 0:
            # Remove N-terminal hydrogens from N-terminus
            residue = residue[
                (residue.atom_name != "H") &
                (residue.atom_name != "H2")
            ]
        if i != len(res_names)-1:
            # Remove C-terminal hydroxyl group
            residue = residue[
                (residue.atom_name != "OXT") &
                (residue.atom_name != "HXT")
            ]
        _, transformation = struc.superimpose(
            backbone_for_res[struc.filter_backbone(backbone_for_res)],
            residue[struc.filter_backbone(residue)]
        )
        
        full_residue = backbone_for_res \
            + struc.superimpose_apply(side_chain, transformation)
        full_residue.chain_id[:] = "A"
        full_residue.res_id[:] = i+1
        full_residue.res_name[:] = res_name
        full_atom_structure += full_residue
    
    return full_atom_structure


sequence = seq.ProteinSequence("TITANITE")
#sequence = seq.ProteinSequence("TITANITE")
atom_array = assemble_peptide(sequence)

strucio.save_structure("bb.mmtf", atom_array)