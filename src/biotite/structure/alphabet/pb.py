# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Conversion of structures into the *Protein Blocks* structural alphabet.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Patrick Kunzmann"
__all__ = ["ProteinBlocksSequence", "to_protein_blocks"]

import numpy as np
from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.sequence import Sequence
from biotite.structure.chains import get_chain_starts
from biotite.structure.geometry import dihedral_backbone

# PB reference angles, adapted from PBxplore
PB_ANGLES = np.array(
    [
        [41.14,    75.53,   13.92,  -99.80,  131.88,  -96.27, 122.08,  -99.68],
        [108.24,  -90.12,  119.54,  -92.21,  -18.06, -128.93, 147.04,  -99.90],
        [-11.61, -105.66,   94.81, -106.09,  133.56, -106.93, 135.97, -100.63],
        [141.98, -112.79,  132.20, -114.79,  140.11, -111.05, 139.54, -103.16],
        [133.25, -112.37,  137.64, -108.13,  133.00,  -87.30, 120.54,   77.40],
        [116.40, -105.53,  129.32,  -96.68,  140.72,  -74.19, -26.65,  -94.51],
        [0.40,    -81.83,    4.91, -100.59,   85.50,  -71.65, 130.78,   84.98],
        [119.14, -102.58,  130.83,  -67.91,  121.55,   76.25,  -2.95,  -90.88],
        [130.68,  -56.92,  119.26,   77.85,   10.42,  -99.43, 141.40,  -98.01],
        [114.32, -121.47,  118.14,   82.88, -150.05,  -83.81,  23.35,  -85.82],
        [117.16,  -95.41,  140.40,  -59.35,  -29.23,  -72.39, -25.08,  -76.16],
        [139.20,  -55.96,  -32.70,  -68.51,  -26.09,  -74.44, -22.60,  -71.74],
        [-39.62,  -64.73,  -39.52,  -65.54,  -38.88,  -66.89, -37.76,  -70.19],
        [-35.34,  -65.03,  -38.12,  -66.34,  -29.51,  -89.10,  -2.91,   77.90],
        [-45.29,  -67.44,  -27.72,  -87.27,    5.13,   77.49,  30.71,  -93.23],
        [-27.09,  -86.14,    0.30,   59.85,   21.51,  -96.30, 132.67,  -92.91],
    ]
)  # fmt: skip


class ProteinBlocksSequence(Sequence):
    """
    Representation of a structure in the *Protein Blocks* structural alphabet.
    :footcite:`Brevern2000`

    Parameters
    ----------
    sequence : iterable object, optional
        The *Protein Blocks* sequence.
        This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.

    See Also
    --------
    to_protein_blocks : Create *Protein Blocks* sequences from a structure.

    References
    ----------

    .. footbibliography::
    """

    alphabet = LetterAlphabet("abcdefghijklmnopz")
    undefined_symbol = "z"

    def __init__(self, sequence=""):
        if isinstance(sequence, str):
            sequence = sequence.lower()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        super().__init__(sequence)

    def get_alphabet(self):
        return ProteinBlocksSequence.alphabet

    def remove_undefined(self):
        """
        Remove undefined symbols from the sequence.

        Returns
        -------
        filtered_sequence : ProteinBlocksSequence
            The sequence without undefined symbols.
        """
        undefined_code = ProteinBlocksSequence.alphabet.encode(
            ProteinBlocksSequence.undefined_symbol
        )
        filtered_code = self.code[self.code != undefined_code]
        filtered_sequence = ProteinBlocksSequence()
        filtered_sequence.code = filtered_code
        return filtered_sequence


def to_protein_blocks(atoms):
    """
    Encode each chain in the given structure to the *Protein Blocks* structural
    alphabet.
    :footcite:`Brevern2000`

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode.
        May contain multiple chains.

    Returns
    -------
    sequences : list of Sequence, length=n
        The encoded *Protein Blocks* sequence for each peptide chain in the structure.
    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> sequences, chain_starts = to_protein_blocks(atom_array)
    >>> print(sequences[0])
    zzmmmmmnopjmnopacdzz
    """
    sequences = []
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        sequences.append(_to_protein_blocks(chain))
    return sequences, chain_start_indices[:-1]


def _to_protein_blocks(chain):
    undefined_code = ProteinBlocksSequence.alphabet.encode(
        ProteinBlocksSequence.undefined_symbol
    )

    phi, psi, _ = dihedral_backbone(chain)

    pb_angles = np.full((len(phi), 8), np.nan)
    pb_angles[2:-2, 0] = psi[:-4]
    pb_angles[2:-2, 1] = phi[1:-3]
    pb_angles[2:-2, 2] = psi[1:-3]
    pb_angles[2:-2, 3] = phi[2:-2]
    pb_angles[2:-2, 4] = psi[2:-2]
    pb_angles[2:-2, 5] = phi[3:-1]
    pb_angles[2:-2, 6] = psi[3:-1]
    pb_angles[2:-2, 7] = phi[4:]
    pb_angles = np.rad2deg(pb_angles)

    # Angle RMSD of all reference angles with all actual angles
    rmsda = np.sum(
        ((PB_ANGLES[:, np.newaxis] - pb_angles[np.newaxis, :] + 180) % 360 - 180) ** 2,
        axis=-1,
    )
    # Where RMSDA is NaN, (missing atoms/residues or chain ends) set symbol to unknown
    pb_seq_code = np.full(len(pb_angles), undefined_code, dtype=np.uint8)
    pb_available_mask = ~np.isnan(rmsda).any(axis=0)
    # Chose PB, where the RMSDA to the reference angle is lowest
    # Due to the definition of Biotite symbol codes
    # the index of the chosen PB is directly the symbol code
    pb_seq_code[pb_available_mask] = np.argmin(rmsda[:, pb_available_mask], axis=0)
    # Put the array of symbol codes into actual sequence objects
    pb_sequence = ProteinBlocksSequence()
    pb_sequence.code = pb_seq_code
    return pb_sequence
