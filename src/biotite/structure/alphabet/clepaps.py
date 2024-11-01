# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Conversion of structures into the *Protein Blocks* structural alphabet.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Patrick Kunzmann"
__all__ = ["ClepapsSequence", "to_clepaps"]

import numpy as np
from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.sequence import Sequence
from biotite.structure.chains import get_chain_starts
from biotite.structure.filter import filter_amino_acids
from biotite.structure.geometry import angle, dihedral
from biotite.structure.util import coord_for_atom_name_per_residue

# CLePAPS reference angles
CLEPAPS_CENTERS = np.array(
    [
       [ 1.02, -2.  ,  1.55],
       [ 1.06, -2.94,  1.34],
       [ 1.01, -1.88,  1.14],
       [ 0.79, -2.3 ,  1.03],
       [ 1.02, -2.98,  0.95],
       [ 1.09, -2.72,  0.91],
       [ 1.49,  2.09,  1.05],
       [ 1.55,  0.88,  1.55],
       [ 1.52,  0.83,  1.52],
       [ 1.58,  1.05,  1.55],
       [ 1.48,  0.7 ,  1.43],
       [ 1.4 ,  0.75,  0.84],
       [ 1.47,  1.64,  1.44],
       [ 1.12,  0.14,  1.49],
       [ 1.54, -1.89,  1.48],
       [ 1.24, -2.98,  1.49],
       [ 0.86, -0.37,  1.01],
    ]
)  # fmt: skip


class ClepapsSequence(Sequence):
    """
    Representation of a structure in the *CLePAPS* structural alphabet.
    :footcite:`Wang2008`

    Parameters
    ----------
    sequence : iterable object, optional
        The *CLePAPS* sequence.
        This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.

    See also
    --------
    to_clepaps : Create *CLePAPS* sequences from a structure.

    References
    ----------

    .. footbibliography::

    """

    alphabet = LetterAlphabet("ABCDEFGHIJKLMNOPQR")
    unknown_symbol = "R"

    def get_alphabet(self):
        return ClepapsSequence.alphabet


def to_clepaps(atoms):
    """
    Encode each chain in the given structure to the *CLePAPS* structural
    alphabet.
    :footcite:`Wang2008`

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode.
        May contain multiple chains.

    Returns
    -------
    sequences : list of Sequence, length=n
        The encoded *CLePAPS* sequence for each peptide chain in the structure.
    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> sequences, chain_starts = to_clepaps(atom_array)
    >>> print(sequences[0])
    """
    sequences = []
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        sequences.append(_to_clepaps(chain))
    return sequences, chain_start_indices[:-1]


def _to_clepaps(chain):
    amino_acid_mask = filter_amino_acids(chain)

    # Coordinates for dihedral angle calculation
    (coord_ca,) = coord_for_atom_name_per_residue(
        chain,
        ("CA",),
        amino_acid_mask,
    )

    bending = angle(coord_ca[:-2], coord_ca[1:-1], coord_ca[2:])
    theta_1 = bending[:-1]
    theta_2 = bending[1:]
    tau = dihedral(coord_ca[:-3], coord_ca[1:-2], coord_ca[2:-1], coord_ca[3:])
    clepaps_angles = np.stack([theta_1, tau, theta_2], axis=-1)

    # Angle RMSD of all reference angles with all actual angles
    rmsda = np.sum(
        (CLEPAPS_CENTERS[:, np.newaxis] - clepaps_angles[np.newaxis, :]) ** 2,
        axis=-1,
    )
    # Where RMSDA is NaN, (missing atoms/residues or chain ends) set symbol to unknown
    clepaps_seq_code = np.full(
        len(clepaps_angles),
        ClepapsSequence.alphabet.encode(ClepapsSequence.unknown_symbol),
    )
    available_mask = ~np.isnan(rmsda).any(axis=0)
    # Chose symbol, where the RMSDA to the reference angle is lowest
    # Due to the definition of Biotite symbol codes
    # the index of the chosen PB is directly the symbol code
    clepaps_seq_code[available_mask] = np.argmin(rmsda[:, available_mask], axis=0)
    # Put the array of symbol codes into actual sequence objects
    clepaps_sequence = ClepapsSequence()
    # Since every symbols comprises 4 residues, the sequence length is shortened by 3
    # By definition of CLePAPS, the first two and the last residue are undefined
    clepaps_sequence.code = np.full(
        coord_ca.shape[0],
        ClepapsSequence.alphabet.encode(ClepapsSequence.unknown_symbol),
    )
    clepaps_sequence.code[2:-1] = clepaps_seq_code
    return clepaps_sequence
