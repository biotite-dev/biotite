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
import biotite.structure.info as info


N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.45
C_N_LENGTH  = 1.34


def assemble_peptide(sequence):
    backbone = struc.AtomArray(length=3*len(sequence))
    for i, bond_length, angle in zip(
        range(backbone.array_length()),
        itertools.cycle([C_N_LENGTH, N_CA_LENGTH, CA_C_LENGTH]),
        itertools.cycle([120, -120])
    ):
        if i == 0:
            backbone.coord[i] = [0,0,0]
        elif i == 1:
            backbone.coord[i] = [bond_length,0,0]


    full_atom = struc.AtomArray(length=0)
    res_names = [seq.ProteinSequence.convert_letter_1to3(r) for r in sequence]
    for residue in res_names:
        print(residue)
    
    return full_atom


sequence = seq.ProteinSequence("TITANITE")
atom_array = assemble_peptide(sequence)