#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
from biopython.structure.io import read_structure_from_file
import glob
from os.path import join
from .util import data_dir
from biopython.structure.atoms import AtomArrayStack, AtomArray

pdb_paths = glob.glob(join(data_dir, "*.pdb"))

@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.pdb")))
def test_assume_fileformat(path):
    struct = read_structure_from_file(path)
    assert isinstance(struct, AtomArrayStack) or isinstance(struct, AtomArray)
    
@pytest.mark.parametrize("path", glob.glob(join(data_dir, "*.pdb")))
def test_load_pdb(path):
    struct = read_structure_from_file(path, format='pdb')
    assert isinstance(struct, AtomArrayStack) or isinstance(struct, AtomArray)