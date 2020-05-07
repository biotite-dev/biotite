import biotite.structure as struc
import biotite.structure.io as strucio
import pytest
from os.path import join
from ..util import data_dir
from biotite.structure.basepairs import _get_proximate_basepair_candidates

def test__get_proximate_basepair_candidates():
    nuc_sample_array = strucio.load_structure(
        join(data_dir("structure"), "5ugo.cif")
    )
    
    assert ( len(_get_proximate_basepair_candidates(nuc_sample_array))
                == 128 )

