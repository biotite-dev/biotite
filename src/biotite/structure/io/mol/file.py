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
from ...bonds import BondType


# Number of header lines
N_HEADER = 3
DATE_FORMAT = "%d%m%y%H%M"


class MOLFile(TextFile):
    """
    This class represents a file in MOL format, that is used to store
    structure information for small molecules. :footcite:`Dalby1992`

    Since its use is intended for single small molecules, it stores
    less atom annotation information than the macromolecular structure
    formats:
    Only the atom positions, charges, elements and bonds can be read
    from the file, chain and and residue information is missing.

    This class can also be used to parse the first structure from an SDF
    file, as the SDF format extends the MOL format.

    References
    ----------

    .. footbibliography::

    Examples
    --------

    >>> from os.path import join
    >>> mol_file = MOLFile.read(join(path_to_structures, "molecules", "TYR.sdf"))
    >>> atom_array = mol_file.get_structure()
    >>> print(atom_array)
                0             N         1.320    0.952    1.428
                0             C        -0.018    0.429    1.734
                0             C        -0.103    0.094    3.201
                0             O         0.886   -0.254    3.799
                0             C        -0.274   -0.831    0.907
                0             C        -0.189   -0.496   -0.559
                0             C         1.022   -0.589   -1.219
                0             C        -1.324   -0.102   -1.244
                0             C         1.103   -0.282   -2.563
                0             C        -1.247    0.210   -2.587
                0             C        -0.032    0.118   -3.252
                0             O         0.044    0.420   -4.574
                0             O        -1.279    0.184    3.842
                0             H         1.977    0.225    1.669
                0             H         1.365    1.063    0.426
                0             H        -0.767    1.183    1.489
                0             H         0.473   -1.585    1.152
                0             H        -1.268   -1.219    1.134
                0             H         1.905   -0.902   -0.683
                0             H        -2.269   -0.031   -0.727
                0             H         2.049   -0.354   -3.078
                0             H        -2.132    0.523   -3.121
                0             H        -0.123   -0.399   -5.059
                0             H        -1.333   -0.030    4.784
    """

    def __init__(self):
        super().__init__()
        # empty header lines
        self.lines = [""] * N_HEADER

    def get_header(self):
        """
        Get the header from the MOL file.

        Returns
        -------
        mol_name : str
            The name of the molecule.
        initials : str
            The author's initials.
        program : str
            The program name.
        time : datetime
            The time of file creation.
        dimensions : str
            Dimensional codes.
        scaling_factors : str
            Scaling factors.
        energy : str
            Energy from modeling program.
        registry_number : str
            MDL registry number.
        comments : str
            Additional comments.
        """
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
        """
        Set the header for the MOL file.

        Parameters
        ----------
        mol_name : str
            The name of the molecule.
        initials : str, optional
            The author's initials. Maximum length is 2.
        program : str, optional
            The program name. Maximum length is 8.
        time : datetime or date, optional
            The time of file creation.
        dimensions : str, optional
            Dimensional codes. Maximum length is 2.
        scaling_factors : str, optional
            Scaling factors. Maximum length is 12.
        energy : str, optional
            Energy from modeling program. Maximum length is 12.
        registry_number : str, optional
            MDL registry number. Maximum length is 6.
        comments : str, optional
            Additional comments.
        """
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
        """
        Get an :class:`AtomArray` from the MOL file.

        Returns
        -------
        array : AtomArray
            This :class:`AtomArray` contains the optional ``charge``
            annotation and has an associated :class:`BondList`.
            All other annotation categories, except ``element`` are
            empty.
        """
        ctab_lines = _get_ctab_lines(self.lines)
        if len(ctab_lines) == 0:
            raise InvalidFileError("File does not contain structure data")
        return read_structure_from_ctab(ctab_lines)


    def set_structure(self, atoms, default_bond_type=BondType.ANY):
        """
        Set the :class:`AtomArray` for the file.

        Parameters
        ----------
        array : AtomArray
            The array to be saved into this file.
            Must have an associated :class:`BondList`.
        default_bond_type : BondType
            Bond type fallback in the *Bond block* if a bond has no bond_type
            defined in *atoms* array. By default, each bond is treated as
            :attr:`BondType.ANY`.
        """
        self.lines = self.lines[:N_HEADER] + write_structure_to_ctab(
            atoms,
            default_bond_type
        )


def _get_ctab_lines(lines):
    for i, line in enumerate(lines):
        if line.startswith("M  END"):
            return lines[N_HEADER:i+1]
    return lines[N_HEADER:]
