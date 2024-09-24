# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
NumPy port of the ``foldseek`` code for encoding structures to 3di.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde"
__all__ = ["I3DSequence", "to_3di"]

import numpy as np
from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.sequence import Sequence
from biotite.structure.alphabet.encoder import Encoder
from biotite.structure.chains import get_chain_starts


class I3DSequence(Sequence):
    """
    Representation of a structure sequence (in 3di alphabet).

    Parameters
    ----------
    sequence : iterable object, optional
        The initial 3Di sequence.
        This may either be a list or a string.
        May take upper or lower case letters.
        By default the sequence is empty.
    """

    alphabet = LetterAlphabet(
        [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
    )
    unknown_symbol = "D"

    def __init__(self, sequence=""):
        if isinstance(sequence, str):
            sequence = sequence.upper()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        seq_code = I3DSequence.alphabet.encode_multiple(sequence)
        super().__init__()
        self.code = seq_code

    def get_alphabet(self):
        return I3DSequence.alphabet

    def __repr__(self):
        return f'I3DSequence("{"".join(self.symbols)}")'


def to_3di(atoms):
    r"""
    Encode each chain in the given structure to the 3Di structure alphabet.

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode. All atoms must be part of
        a single chain.
        May contain multiple chains.

    Returns
    -------
    sequences : list of Sequence, length=n
        The encoded 3Di sequence for each peptide chain in the structure.

    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.
    """
    sequences = []
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        sequence = I3DSequence()
        sequence.code = _encode_atoms(chain).filled()
        sequences.append(sequence)
    return sequences, chain_start_indices[:-1]


def _encode_atoms(atoms):
    ca_atoms = atoms[atoms.atom_name == "CA"]
    cb_atoms = atoms[atoms.atom_name == "CB"]
    n_atoms = atoms[atoms.atom_name == "N"]
    c_atoms = atoms[atoms.atom_name == "C"]

    r = atoms.res_id.max()

    ca = np.zeros((r + 1, 3), dtype=np.float32)
    ca.fill(np.nan)
    cb = ca.copy()
    n = ca.copy()
    c = ca.copy()

    ca[ca_atoms.res_id, :] = ca_atoms.coord
    cb[cb_atoms.res_id, :] = cb_atoms.coord
    n[n_atoms.res_id, :] = n_atoms.coord
    c[c_atoms.res_id, :] = c_atoms.coord

    ca = ca[ca_atoms.res_id]
    cb = cb[ca_atoms.res_id]
    n = n[ca_atoms.res_id]
    c = c[ca_atoms.res_id]

    return Encoder().encode(ca, cb, n, c)
