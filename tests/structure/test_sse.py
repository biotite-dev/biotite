# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join, basename, splitext
import numpy as np
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.sequence.io.fasta as fasta
from ..util import data_dir


def test_sse():
    """
    Compare output of :func:`annotate_sse()` to output of original
    *P-SEA* software and ensure that almost all annotations are equal.
    Small deviations may appear due to some ambiguous descriptions in
    the original algorithm description.
    """
    THRESHOLD = 0.85

    matches = 0
    total = 0

    for file_name in glob.glob(join(data_dir("structure"), "*.mmtf")):

        pdb_id = splitext(basename(file_name))[0]

        atoms = mmtf.get_structure(mmtf.MMTFFile.read(file_name), model=1)
        atoms = atoms[struc.filter_canonical_amino_acids(atoms)]
        if atoms.array_length() == 0:
            # Structure contains no peptide to annotate SSE for
            continue
        atoms = atoms[atoms.chain_id == atoms.chain_id[0]]

        ref_sse = fasta.FastaFile.read(
            join(data_dir("structure"), "psea.fasta")
        )[pdb_id]
        ref_sse = np.array(list(ref_sse))
        
        test_sse = struc.annotate_sse(atoms, atoms.chain_id[0])
        
        assert len(test_sse) == len(ref_sse)
        matches += np.count_nonzero(test_sse == ref_sse)
        total += len(ref_sse)

    assert matches / total >= THRESHOLD