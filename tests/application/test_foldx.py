# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.info as info
import biotite.structure.io.mmtf as mmtf
from biotite.application.foldx import FoldXApp
from biotite.structure.io.pdb import PDBFile
from ..util import data_dir, is_not_installed


@pytest.mark.skipif(
    is_not_installed("foldx"), reason="FoldX is not installed"
)

def test_folding():
    """
    Test :class:`FoldXApp` check mutation to streptavidin.
    The output binding pose should be very similar to the pose in the
    PDB structure.
    """
    # A structure of a straptavidin-biotin complex
    mmtf_file = mmtf.MMTFFile.read(join(data_dir("application"), "2rtg.mmtf"))
    structure = mmtf.get_structure(
        mmtf_file, model=1, extra_fields=["charge"], include_bonds=True
    )
    structure = structure[structure.chain_id == "B"]
    receptor = structure[struc.filter_amino_acids(structure)]
    mutation = "A13A"

    app = FoldXApp(receptor, mutation, subunit = 'B')
    app.start()
    app.join()
    arr = app.get_mutant()
    file = PDBFile()
    assert (file is not None)