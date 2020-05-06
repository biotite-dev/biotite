from os.path import join
import itertools
import numpy as np
import pytest
import biotite.structure as struc
import biotite.structure.io as strucio
from ..util import data_dir
import biotite

def test___get_proximate_basepair_candidates__():
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "5ugo.cif")
    )
    
    print(len(struc.__get_proximate_basepair_candidates__(nuc_sample_array)))

    assert False
