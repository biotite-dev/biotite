# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.mark.parametrize("path", glob.glob(join(data_dir("structure"), "*.bcif")))
def test_pdbx_sequence_consistency(path):
    """
    Check if sequences created with :func:`to_sequence()` are equal to
    the ones already stored in the PDBx file.
    """
    if "4gxy" in path:
        pytest.skip(
            "Edge case: contains 'GTP' which has one-letter code, "
            "but is a 'NON-POLYMER' in the CCD"
        )
    if "4zxb" in path:
        pytest.skip(
            "Chain C residues are all standard but fails due to too many missing"
        )

    pdbx_file = pdbx.BinaryCIFFile.read(path)
    ref_sequences = pdbx.get_sequence(pdbx_file)

    atoms = pdbx.get_structure(pdbx_file, use_author_fields=False, model=1)
    # Remove pure solvent chains
    # In those chains the "label_seq_id" is usually "."
    # which is translated to -1
    atoms = atoms[atoms.res_id != -1]
    test_sequences, _ = struc.to_sequence(atoms, allow_hetero=True)

    # Matching against the PDBx file is not trivial
    #   * The file stores duplicate sequences only once
    #     -> Matching any sequence from file is sufficient
    #   * The resolved part of the chain might miss residues
    #     -> Sequence identity from alignment is expected to be 1.0
    for sequence in test_sequences:
        best_aln, best_identity = _find_best_match(sequence, ref_sequences)
        try:
            assert best_identity == 1.0
        except AssertionError:
            print(best_aln)
            raise


def _find_best_match(sequence, ref_sequences):
    best_alignment = None
    best_identity = 0.0
    for ref_sequence in ref_sequences.values():
        if not isinstance(sequence, type(ref_sequence)):
            continue
        if isinstance(sequence, seq.ProteinSequence):
            matrix = align.SubstitutionMatrix.std_protein_matrix()
        else:
            matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
        alignment = align.align_optimal(
            sequence, ref_sequence, matrix, terminal_penalty=False, max_number=1
        )[0]
        # The 'shortest' identity is 1.0, if every residue in the
        # test sequence is aligned to an identical residue
        identity = align.get_sequence_identity(alignment, mode="shortest")
        if identity > best_identity:
            best_alignment = alignment
            best_identity = identity
    return best_alignment, best_identity
