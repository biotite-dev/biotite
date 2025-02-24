# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.io.mol"
__author__ = "Patrick Kunzmann"
__all__ = ["MOLFile"]

from biotite.file import InvalidFileError, TextFile
from biotite.structure.bonds import BondType
from biotite.structure.io.mol.ctab import (
    read_structure_from_ctab,
    write_structure_to_ctab,
)
from biotite.structure.io.mol.header import Header

# Number of header lines
N_HEADER = 3


class MOLFile(TextFile):
    """
    This class represents a file in MOL format, that is used to store
    structure information for small molecules.
    :footcite:`Dalby1992`

    Since its use is intended for single small molecules, it stores
    less atom annotation information than the macromolecular structure
    formats:
    Only the atom positions, charges, elements and bonds can be read
    from the file, chain and and residue information is missing.

    This class can also be used to parse the first structure from an SDF
    file, as the SDF format extends the MOL format.

    Attributes
    ----------
    header : Header
        The header of the MOL file.

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
        self._header = None

    @classmethod
    def read(cls, file):
        mol_file = super().read(file)
        mol_file._header = None
        return mol_file

    @property
    def header(self):
        if self._header is None:
            self._header = Header.deserialize("\n".join(self.lines[0:3]) + "\n")
        return self._header

    @header.setter
    def header(self, header):
        self._header = header
        self.lines[0:3] = self._header.serialize().splitlines()

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

    def set_structure(self, atoms, default_bond_type=BondType.ANY, version=None):
        """
        Set the :class:`AtomArray` for the file.

        Parameters
        ----------
        atoms : AtomArray
            The array to be saved into this file.
            Must have an associated :class:`BondList`.
        default_bond_type : BondType, optional
            Bond type fallback for the *Bond block*, if a
            :class:`BondType` has no CTAB counterpart.
            By default, each such bond is treated as
            :attr:`BondType.ANY`.
        version : {"V2000", "V3000"}, optional
            The version of the CTAB format.
            ``"V2000"`` uses the *Atom* and *Bond* block, while
            ``"V3000"`` uses the *Properties* block.
            By default, ``"V2000"`` is used, unless the number of atoms
            or bonds exceeds 999, in which case ``"V3000"`` is used.
        """
        self.lines = self.lines[:N_HEADER] + write_structure_to_ctab(
            atoms, default_bond_type, version
        )


def _get_ctab_lines(lines):
    for i, line in enumerate(lines):
        if line.startswith("M  END"):
            return lines[N_HEADER : i + 1]
    return lines[N_HEADER:]
