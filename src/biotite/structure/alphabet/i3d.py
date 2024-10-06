# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
NumPy port of the ``foldseek`` code for encoding structures to 3di.
"""

__name__ = "biotite.structure.alphabet"
__author__ = "Martin Larralde"
__all__ = ["I3DSequence", "to_3di"]

from biotite.sequence.alphabet import LetterAlphabet
from biotite.sequence.sequence import Sequence
from biotite.structure.alphabet.encoder import Encoder
from biotite.structure.error import BadStructureError
import numpy as np


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

    def __init__(self, sequence=""):
        if isinstance(sequence, str):
            sequence = sequence.upper()
        else:
            sequence = [symbol.upper() for symbol in sequence]
        self._alphabet = I3DSequence.alphabet
        seq_code = self._alphabet.encode_multiple(sequence)
        super().__init__()
        self.code = seq_code

    def get_alphabet(self):
        return I3DSequence.alphabet

    def __repr__(self):
        return f'I3DSequence("{"".join(self.symbols)}")'


def to_3di(atoms):
    r"""
    Encode the atoms to the 3di structure alphabet.

    Parameters
    ----------
    atoms : AtomArray
        The atom array to encode. All atoms must be part of
        a single chain.

    Returns
    -------
    sequence : I3DSequence
        The encoded 3di sequence.

    Note
    ----
    To encode atoms in different chains, use :func:`apply_chain_wise` to
    return a list with one sequence per chain.
    """

    sequence = I3DSequence()
    sequence.code = _encode_atoms(atoms).filled()

    return sequence


def _encode_atoms(
        atoms
    ):
        if not np.all(atoms.chain_id == atoms.chain_id[0]):
            raise BadStructureError("Structure contains more than one chain")

        ca_atoms = atoms[atoms.atom_name == 'CA']
        cb_atoms = atoms[atoms.atom_name == 'CB']
        n_atoms = atoms[atoms.atom_name == 'N']
        c_atoms = atoms[atoms.atom_name == 'C']

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