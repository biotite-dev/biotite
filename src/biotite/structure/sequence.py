# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Function for converting a structure into a sequence.
"""

__name__ = "biotite.structure"
__author__ = "Patrick Kunzmann"
__all__ = ["to_sequence"]

import numpy as np
from biotite.sequence.seqtypes import NucleotideSequence, ProteinSequence
from biotite.structure.chains import get_chain_starts
from biotite.structure.error import BadStructureError
from biotite.structure.info.groups import amino_acid_names, nucleotide_names
from biotite.structure.info.misc import one_letter_code
from biotite.structure.residues import get_residues

HETERO_PLACEHOLDER = "."


def to_sequence(atoms, allow_hetero=False):
    """
    Convert each chain in a structure into a sequence.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure.
        May contain multiple chains.
        Each chain must be either a peptide or a nucleic acid.
    allow_hetero : bool, optional
        If true, residues inside a amino acid or nucleotide chain,
        that have no one-letter code, are replaced by the respective
        '*any*' symbol (`"X"` or `"N"`, respectively).
        The same is true for amino acids in nucleotide chains and vice
        versa.
        By default, an exception is raised.

    Returns
    -------
    sequences : list of Sequence, length=n
        The sequence for each chain in the structure.
    chain_start_indices : ndarray, shape=(n,), dtype=int
        The atom index where each chain starts.

    Notes
    -----
    Residues are considered amino acids or nucleotides based on their
    appearance :func:`info.amino_acid_names()` or
    :func:`info.nucleotide_names()`, respectively.

    Examples
    --------

    >>> sequences, chain_starts = to_sequence(atom_array)
    >>> print(sequences)
    [ProteinSequence("NLYIQWLKDGGPSSGRPPPS")]
    """
    sequences = []
    chain_start_indices = get_chain_starts(atoms, add_exclusive_stop=True)
    for i in range(len(chain_start_indices) - 1):
        start = chain_start_indices[i]
        stop = chain_start_indices[i + 1]
        chain = atoms[start:stop]
        _, residues = get_residues(chain)
        one_letter_symbols = np.array(
            [one_letter_code(res) or HETERO_PLACEHOLDER for res in residues]
        )
        hetero_mask = one_letter_symbols == HETERO_PLACEHOLDER

        aa_count = np.count_nonzero(np.isin(residues, amino_acid_names()))
        nuc_count = np.count_nonzero(np.isin(residues, nucleotide_names()))
        if aa_count == 0 and nuc_count == 0:
            raise BadStructureError(
                f"Chain {chain.chain_id[0]} contains neither amino acids "
                "nor nucleotides"
            )
        elif aa_count > nuc_count:
            # Chain is a peptide
            hetero_mask |= ~np.isin(residues, amino_acid_names())
            if not allow_hetero and np.any(hetero_mask):
                hetero_indices = np.where(hetero_mask)[0]
                raise BadStructureError(
                    f"Hetero residue(s) "
                    f"{', '.join(residues[hetero_indices])} in peptide"
                )
            one_letter_symbols[hetero_mask] = "X"
            # Replace selenocysteine and pyrrolysine
            one_letter_symbols[one_letter_symbols == "U"] = "C"
            one_letter_symbols[one_letter_symbols == "O"] = "K"
            sequences.append(ProteinSequence("".join(one_letter_symbols)))
        else:
            # Chain is a nucleic acid
            hetero_mask |= ~np.isin(residues, nucleotide_names())
            if not allow_hetero and np.any(hetero_mask):
                hetero_indices = np.where(hetero_mask)[0]
                raise BadStructureError(
                    f"Hetero residue(s) "
                    f"{', '.join(residues[hetero_indices])} in nucleic acid"
                )
            one_letter_symbols[hetero_mask] = "N"
            # Replace uracil
            one_letter_symbols[one_letter_symbols == "U"] = "T"
            sequences.append(NucleotideSequence("".join(one_letter_symbols)))

    # Remove exclusive stop
    return sequences, chain_start_indices[:-1]
