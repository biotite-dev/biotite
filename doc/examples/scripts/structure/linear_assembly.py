r"""
Ab initio assembly of a linear peptide
======================================
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import itertools
import numpy as np
from numpy.linalg import norm
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.info as info



N_CA_LENGTH        = 1.46
CA_C_LENGTH        = 1.54
C_N_LENGTH         = 1.34
C_O_LENGTH         = 1.43
C_O_DOUBLE_LENGTH  = 1.23
N_H_LENGTH         = 1.01
O_H_LENGTH         = 0.97

C_O_H_ANGLE = 109



def calculate_atom_coord_by_z_rotation(coord1, coord2, angle, bond_length):
    rot_axis = [0, 0, 1]

    # Calculate the coordinates of a new atoms by rotating the previous
    # bond by the given angle (usually 120 degrees) 
    new_coord = struc.rotate_about_axis(
            atoms = coord2,
            axis = rot_axis,
            angle = np.deg2rad(angle),
            support = coord1
        )
    
    # Scale bond to correct bond length
    bond_vector = new_coord - coord1
    new_coord = coord1 + bond_vector * bond_length / norm(bond_vector)

    return new_coord



def assemble_peptide(sequence):
    res_names = [seq.ProteinSequence.convert_letter_1to3(r) for r in sequence]
    peptide = struc.AtomArray(length=0)
    rot_axis = [0, 0, 1]
    

    for res_id, res_name, connect_angle in zip(np.arange(1, len(res_names)+1),
                                               res_names,
                                               itertools.cycle([120, -120])):
        # Create backbone
        atom_n = struc.Atom(
            [0.0, 0.0, 0.0], atom_name="N", element="N"
        )
        
        atom_ca = struc.Atom(
            [0.0, N_CA_LENGTH, 0.0], atom_name="CA", element="C"
        )
        
        coord_c = calculate_atom_coord_by_z_rotation(
            atom_ca.coord, atom_n.coord, 120, CA_C_LENGTH
        )
        atom_c = struc.Atom(
            coord_c, atom_name="C", element="C"
        )
        
        coord_o = calculate_atom_coord_by_z_rotation(
            atom_c.coord, atom_ca.coord, 120, C_O_DOUBLE_LENGTH
        )
        atom_o = struc.Atom(
            coord_o, atom_name="O", element="O"
        )
        
        coord_h = calculate_atom_coord_by_z_rotation(
            atom_n.coord, atom_ca.coord, -120, N_H_LENGTH
        )
        atom_h = struc.Atom(
            coord_h, atom_name="H", element="H"
        )

        backbone = struc.array([atom_n, atom_ca, atom_c, atom_o, atom_h])
        backbone.res_id[:] = res_id
        backbone.res_name[:] = res_name
        
        
        # Connect backbone to existing residues in the chain
        if res_id > 1:
            prev_res = peptide[peptide.res_id == res_id-1]
            prev_coord_ca = prev_res[prev_res.atom_name == "CA"][0].coord
            prev_coord_c  = prev_res[prev_res.atom_name == "C" ][0].coord
            
            curr_coord_n  = calculate_atom_coord_by_z_rotation(
                prev_coord_c, prev_coord_ca, connect_angle, C_N_LENGTH
            )
            backbone.coord -= atom_n.coord
            backbone.coord += curr_coord_n
            
            # Adjacent residues should show in opposing directions
            # -> rotate residues with even residue ID by 180 degrees
            if res_id % 2 == 0:
                coord_n = backbone[backbone.atom_name == "N"][0].coord
                coord_c = backbone[backbone.atom_name == "C"][0].coord
                backbone = struc.rotate_about_axis(
                    atoms = backbone,
                    axis = coord_c - coord_n,
                    angle = np.deg2rad(180),
                    support = coord_n
                )


        # Get residue from dataset
        residue = info.residue(res_name)
        # Superimpose backbone of residue
        # with backbone created previously 
        _, transformation = struc.superimpose(
            backbone[struc.filter_backbone(backbone)],
            residue[struc.filter_backbone(residue)]
        )
        residue = struc.superimpose_apply(residue, transformation)
        # Remove backbone atoms from residue because they are already
        # existing in the backbone created prevoisly
        side_chain = residue[~np.isin(
            residue.atom_name,
            ["N", "CA", "C", "O", "OXT", "H", "H2", "H3", "HXT"]
        )]
        

        # Assemble backbone with side chain (including HA)
        # and set annotation arrays
        residue = backbone + side_chain
        residue.chain_id[:] = "A"
        residue.res_id[:] = res_id
        residue.res_name[:] = res_name
        peptide += residue
    

    # Add N-terminal hydrogen
    atom_n = peptide[(peptide.res_id == 1) & (peptide.atom_name == "N")][0]
    atom_h = peptide[(peptide.res_id == 1) & (peptide.atom_name == "H")][0]
    coord_h2 = calculate_atom_coord_by_z_rotation(
        atom_n.coord, atom_h.coord, -120, N_H_LENGTH
    )
    atom_h2 = struc.Atom(
        coord_h2,
        chain_id="A", res_id=1, res_name=atom_h.res_name, atom_name="H2",
        element="H"
    )
    peptide = struc.array([atom_h2]) + peptide

    # Add C-terminal hydroxyl group
    last = len(sequence)
    atom_c = peptide[(peptide.res_id == last) & (peptide.atom_name == "C")][0]
    atom_o = peptide[(peptide.res_id == last) & (peptide.atom_name == "O")][0]
    coord_oxt = calculate_atom_coord_by_z_rotation(
        atom_c.coord, atom_o.coord, -120, C_O_LENGTH
    )
    coord_hxt = calculate_atom_coord_by_z_rotation(
        coord_oxt, atom_c.coord, np.deg2rad(C_O_H_ANGLE), O_H_LENGTH
    )
    atom_oxt = struc.Atom(
        coord_oxt,
        chain_id="A", res_id=last, res_name=atom_c.res_name, atom_name="OXT",
        element="O"
    )
    atom_hxt = struc.Atom(
        coord_hxt,
        chain_id="A", res_id=last, res_name=atom_c.res_name, atom_name="HXT",
        element="H"
    )
    peptide = peptide + struc.array([atom_oxt, atom_hxt])


    return peptide



#sequence = seq.ProteinSequence("TIT")
sequence = seq.ProteinSequence("TITANITE")
atom_array = assemble_peptide(sequence)
strucio.save_structure("linear.cif", atom_array)