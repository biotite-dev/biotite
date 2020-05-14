import numpy as np
import biotite.structure as struc
import biotite.structure.io as strucio
import pytest
from os.path import join
from ..util import data_dir
from biotite.structure.basepairs import _get_proximate_basepair_candidates, get_basepairs


#TODO: Remove tests for private functions

def test_get_proximate_basepair_candidates():
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "5ugo.cif")
    )
    
    assert ( len(_get_proximate_basepair_candidates(nuc_sample_array))
                == 128 )

def test_get_basepairs():
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "1qxb.cif")
    )
    #nuc_sample_array = nuc_sample_array[np.isin(nuc_sample_array.element, ["H"], invert=True)]
    #print(nuc_sample_array)
    basepairs = [[2, 'A', 11, 'B'], [3, 'A', 10, 'B'], [4, 'A', 9, 'B'],
                 [5, 'A', 8, 'B']
                ]
    print(get_basepairs(nuc_sample_array))
    assert ( get_basepairs(nuc_sample_array) == basepairs )
