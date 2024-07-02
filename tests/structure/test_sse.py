# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.sequence.io.fasta as fasta
from ..util import data_dir


def test_sse():
    """
    Compare output of :func:`annotate_sse()` to output of original
    *P-SEA* software and ensure that most annotations are equal.
    Small deviations may appear due to some ambiguity in the original
    algorithm description.
    """
    THRESHOLD = 0.85

    matches = 0
    total = 0
    ref_psea_file = fasta.FastaFile.read(
        join(data_dir("structure"), "psea.fasta")
    )

    for pdb_id in ref_psea_file:
        ref_sse = np.array(list(ref_psea_file[pdb_id]))

        atoms = pdbx.get_structure(
            pdbx.BinaryCIFFile.read(
                join(data_dir("structure"), f"{pdb_id}.bcif")
            ),
            model=1
        )
        atoms = atoms[struc.filter_canonical_amino_acids(atoms)]
        if atoms.array_length() == 0:
            # Structure contains no peptide to annotate SSE for
            continue
        atoms = atoms[atoms.chain_id == atoms.chain_id[0]]
        test_sse = struc.annotate_sse(atoms)

        assert len(test_sse) == len(ref_sse)
        matches += np.count_nonzero(test_sse == ref_sse)
        total += len(ref_sse)

    assert matches / total >= THRESHOLD


np.random.seed(0)
@pytest.mark.parametrize(
    "discont_pos", np.random.randint(2, 105, size=100)
)
def test_sse_discontinuity(discont_pos):
    """
    Check if discontinuities are properly handled by inserting a
    discontinuity at a random location and expect that the SSE in its
    proximity becomes 'coil'.
    """
    atoms = pdbx.get_structure(
        pdbx.BinaryCIFFile.read(join(data_dir("structure"), "1gya.bcif")),
        model=1
    )
    atoms = atoms[struc.filter_canonical_amino_acids(atoms)]

    ref_sse = struc.annotate_sse(atoms)

    # Initially there should be no discontinuity
    assert len(struc.check_res_id_continuity(atoms)) == 0
    # Introduce discontinuity
    res_starts = struc.get_residue_starts(atoms)
    atoms.res_id[res_starts[discont_pos]:] += 1
    test_sse = struc.annotate_sse(atoms)

    assert len(test_sse) == len(ref_sse)
    # The SSE should be equal everywhere,
    # except in proximity of the discontinuity
    discont_proximity = np.zeros(len(ref_sse), dtype=bool)
    # Use great range in proximity, as the discontinuity may shorten
    # helix/strand region, so that the entire region becomes 'coil'
    discont_proximity[discont_pos - 10 : discont_pos + 10] = True
    assert (test_sse[~discont_proximity] == ref_sse[~discont_proximity]).all()
    # In proximity of the discontinuity we expect 'coil'
    discont_proximity = np.zeros(len(ref_sse), dtype=bool)
    discont_proximity[discont_pos - 2 : discont_pos + 1] = True
    assert (test_sse[discont_proximity] == "c").all()


@pytest.mark.parametrize(
    "file_name", glob.glob(join(data_dir("structure"), "*.bcif"))
)
def test_sse_non_peptide(file_name):
    """
    Test whether only amino acids get SSE annotated.
    """
    atoms = pdbx.get_structure(pdbx.BinaryCIFFile.read(file_name), model=1)

    # Special case for PDB 5EIL:
    # The residue BP5 is an amino acid, but has no CA
    # -> rename analogous atom
    atoms.atom_name[
        (atoms.res_name == "BP5") & (atoms.atom_name == "C13")
    ] = "CA"

    sse = struc.annotate_sse(atoms)
    peptide_mask = struc.filter_amino_acids(atoms)
    # Project mask to residue level
    peptide_mask = peptide_mask[struc.get_residue_starts(atoms)]

    assert np.all(np.isin(sse[peptide_mask], ["a", "b", "c"]))
    assert np.all(sse[~peptide_mask] == "")