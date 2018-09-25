# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["atom_masses"]

import json
from os.path import join, dirname, realpath


_info_dir = dirname(realpath(__file__))

# Masses are taken from http://www.sbcs.qmul.ac.uk/iupac/AtWt/ (2018/03/01)
with open(join(_info_dir, "atom_masses.json")) as file:
    atom_masses = json.load(file)