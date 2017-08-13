# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import requests
import os.path
import os
import glob
import numpy as np

__all__ = ["fetch"]


_dowload_url = "http://ftp.ncbi.nih.gov/blast/matrices/"

def fetch(matrix_name, overwrite=False, quiet=True):
    r = requests.get(_dowload_url + matrix_name)
    content = r.text
    
    lines = content.split("\n")
    lines = [line for line in lines if len(line) != 0 and line[0] != "#"]
    symbols1 = [line[0] for line in lines[1:]]
    symbols2 = [e for e in lines[0].split()]
    scores = np.array([line.split()[1:] for line in lines[1:]]).astype(float)
    scores = np.transpose(scores)
    
    matrix_dict = {}
    for i in range(len(symbols1)):
        for j in range(len(symbols2)):
            matrix_dict[(symbols1[i], symbols1[j])] = scores[i,j]
    return matrix_dict