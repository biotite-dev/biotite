# Copyright 2017 Patrick Kunzmann.
# This source code is part of the Biotite package and is distributed under the
# 3-Clause BSD License.  Please see 'LICENSE.rst' for further information.

import biotite.structure as struc
import biotite.structure.io.npz as npz
import numpy as np
from os.path import join
from .util import data_dir
import pytest

@pytest.fixture
def sample_array():
    file = npz.NpzFile()
    file.read(join(data_dir, "3o5r.npz"))
    return file.get_structure()[0]

def test_solvent_filter(sample_array):
    assert len(sample_array[struc.filter_solvent(sample_array)]) == 287

def test_amino_acid_filter(sample_array):
    assert len(sample_array[struc.filter_amino_acids(sample_array)]) == 982

def test_backbone_filter(sample_array):
    assert len(sample_array[struc.filter_backbone(sample_array)]) == 384

def test_intersection_filter(sample_array):
    assert len(sample_array[:200][
               struc.filter_intersection(sample_array[:200],sample_array[100:])
           ]) == 100