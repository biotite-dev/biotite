"""
Creation of an amino acid rotamer library
=========================================
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import numpy as np
import pyximport
pyximport.install(
    setup_args={'include_dirs': np.get_include()},
    language_level=3
)

import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.info as info


# 'CA' is not in backbone,
# as we want to include the rotation between 'CA' and 'CB'
BACKBONE = ["N", "C", "O", "OXT"]
LIB_SIZE = 1


residue = info.residue("ALA")
bond_list = residue.bonds


# Identify rotatable bonds
rotatable_bonds = []
for atom1, atom2, bond_type in bond_list.as_array():
    # Can only rotate about single bonds
    if bond_type != struc.BondType.SINGLE:
        continue
    
    segmented = bond_list.copy()
    segmented.remove_bond(atom1, atom2)
    conn_atoms1 = struc.find_connected(segmented, atom1)
    conn_atoms2 = struc.find_connected(segmented, atom2)

    # Rotation makes no sense if one of the atoms is a terminal atom,
    # e.g. a hydrogen
    if len(conn_atoms1) == 1 or len(conn_atoms2) == 1:
            continue

    # A bond cannot be rotated in a trivial way,
    # if it is inside a circular structure
    is_circular = atom2 in struc.find_connected(segmented, atom1)
    if is_circular:
        continue

    # Do not rotate about backbone bonds,
    # as these are irrelevant for a rotamer library
    if residue.atom_name[atom1] in BACKBONE or \
       residue.atom_name[atom2] in BACKBONE:
            continue

    # If all consitions pass, add this bond to the rotatable bonds
    rotatable_bonds.append((atom1, atom2, conn_atoms1, conn_atoms2))


print("Rotatable bonds:")
for atom1, atom2, _, _ in rotatable_bonds:
    print(residue.atom_name[atom1] + " <-> " + residue.atom_name[atom2])


# VdW radii of each atom, required for the next step
vdw_radii = np.zeros(residue.array_length())
for i, element in enumerate(residue.element):
    vdw_radii[i] = info.vdw_radius_single(element)
# Pairwise VdW radii sum
vdw_radii_sum = vdw_radii[:, np.newaxis] + vdw_radii[np.newaxis, :]


# Rotate randomly about bonds
print(residue)
np.random.seed(0)
rotamer_coord = np.zeros((LIB_SIZE, residue.array_length(), 3))
for i in range(LIB_SIZE):
    # Coordinates for the current rotamer model
    coord = residue.coord.copy()
    for atom1, atom2, conn_atoms1, conn_atoms2 in rotatable_bonds:
        accepted = False
        while not accepted:
            # A random angle between 0 and 360 degrees
            angle = np.random.rand() * 2*np.pi
            # The bond axis
            axis = coord[atom2] - coord[atom1]
            # Position of one of the involved atoms
            support = coord[i, atom1]
            # Rotate
            coord[conn_atoms1] = struc.rotate_about_axis(
                coord[conn_atoms1], axis, angle, support
            )

            # Check if the atoms clash with each other:
            # The distance between each pair of atoms must be larger
            # than the sum of their VdW radii, if they are not bonded to
            # each other
            accepted = True
            distances = struc.distance(
                coord[:, np.newaxis], coord[np.newaxis, :]
            )
            clashed = distances < vdw_radii_sum
            for clash_atom1, clash_atom2 in zip(*np.where(clashed)):
                if clash_atom1 == clash_atom2:
                    continue
                if (clash_atom1, clash_atom2) not in bond_list:
                    print(clash_atom1, clash_atom2)
                    print(distances[clash_atom1, clash_atom2])
                    print(vdw_radii_sum[clash_atom1, clash_atom2])
                    exit()
                    # Nonbonded atoms clash
                    # -> structure is not accepted
                    accepted = False
        rotamer_coord[i] = coord
rotamers = struc.from_template(residue, rotamer_coord)


# Superimpose backbone onto first model for better visualization
rotamers, _ = struc.superimpose(
    rotamers[0], rotamers, atom_mask=struc.filter_backbone(rotamers)
)


# Write rotamers to structure file
strucio.save_structure("rotamers.mmtf", rotamers)