from os.path import join
import itertools
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from ..util import data_dir
import biotite

import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb

def test_get_proximate_basepair_candidates():
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "5ugo.cif")
    )
    
    print(struc.get_proximate_basepair_candidates(nuc_sample_array))

    assert false
