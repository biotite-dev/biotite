# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"
__all__ = ["MOLFile"]

import datetime
from warnings import warn
import numpy as np
from ...atoms import AtomArray
from ....file import TextFile, InvalidFileError
from ...error import BadStructureError
from ..ctab import read_structure_from_ctab, write_structure_to_ctab


# Number of header lines
N_HEADER = 3
DATE_FORMAT = "%d%m%y%H%M"


class MOLFile(TextFile):
    
    def __init__(self):
        super().__init__()
        # empty header lines
        self.lines = [""] * N_HEADER
    
    def get_header(self):
        mol_name        = self.lines[0].strip()
        initials        = self.lines[1][0:2].strip()
        program         = self.lines[1][2:10].strip()
        time            = datetime.datetime.strptime(self.lines[1][10:20],
                                                     DATE_FORMAT)
        dimensions      = self.lines[1][20:22].strip()
        scaling_factors = self.lines[1][22:34].strip()
        energy          = self.lines[1][34:46].strip()
        registry_number = self.lines[1][46:52].strip()
        comments        = self.lines[2].strip()
        return mol_name, initials, program, time, dimensions, \
               scaling_factors, energy, registry_number, comments

    def set_header(self, mol_name, initials="", program="", time=None,
                   dimensions="", scaling_factors="", energy="",
                   registry_number="", comments=""):
        if time is None:
            time = datetime.datetime.now()
        time_str = time.strftime(DATE_FORMAT)

        self.lines[0] = str(mol_name)
        self.lines[1] = (
            f"{initials:>2}"
            f"{program:>8}"
            f"{time_str:>10}"
            f"{dimensions:>2}"
            f"{scaling_factors:>12}"
            f"{energy:>12}"
            f"{registry_number:>6}"
        )
        self.lines[2] = str(comments)

    def get_structure(self):
        ctab_lines = _get_ctab_lines(self.lines)
        if len(ctab_lines) == 0:
            raise InvalidFileError("File does not contain structure data") 
        return read_structure_from_ctab(ctab_lines)
    
    def set_structure(self, atoms):
        self.lines = self.lines[:N_HEADER] + write_structure_to_ctab(atoms)


def _get_ctab_lines(lines):
    for i, line in enumerate(lines):
        if line.startswith("M  END"):
            return lines[N_HEADER:i+1]
    return lines[N_HEADER:]