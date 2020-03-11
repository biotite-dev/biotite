# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["residue"]

from os.path import join, dirname, realpath
import msgpack
import numpy as np
from ..atoms import AtomArray
from ..bonds import BondList


_residues = None


def _init_dataset():
    """
    Load the residue dataset from MessagePack file.

    Since loading the database is computationally expensive,
    this is only done, when the residue database is actually required.
    """
    global _residues
    if _residues is not None:
        # Database is already initialized
        return

    # Residue data is taken from
    # ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
    # (2019/01/27)
    _info_dir = dirname(realpath(__file__))
    with open(join(_info_dir, "residues.msgpack"), "rb") as file:
        _residues = msgpack.unpack(
            file, use_list=False, raw=False
        )


def residue(res_name):
    """
    Get an atom array, representing the residue with the given name.

    This atom array includes proper values for the ``residue_name``,
    ``hetero``, ``atom_name``, ``element`` and ``charge`` annotation
    arrays and bonds and coordinates.

    Parameters
    ----------
    res_name : str
        The up to 3-letter name of the residue.

    Returns
    -------
    atom_array : AtomArray
        The atom array described by `res_name`.

    Examples
    --------

    >>> alanine = residue("ALA")
    >>> # Atoms and geometry
    >>> print(alanine)
                0  ALA N      N        -0.966    0.493    1.500
                0  ALA CA     C         0.257    0.418    0.692
                0  ALA C      C        -0.094    0.017   -0.716
                0  ALA O      O        -1.056   -0.682   -0.923
                0  ALA CB     C         1.204   -0.620    1.296
                0  ALA OXT    O         0.661    0.439   -1.742
                0  ALA H      H        -1.383   -0.425    1.482
                0  ALA H2     H        -0.676    0.661    2.452
                0  ALA HA     H         0.746    1.392    0.682
                0  ALA HB1    H         1.459   -0.330    2.316
                0  ALA HB2    H         0.715   -1.594    1.307
                0  ALA HB3    H         2.113   -0.676    0.697
                0  ALA HXT    H         0.435    0.182   -2.647
    >>> # Bonds
    >>> print(alanine.atom_name[alanine.bonds.as_array()[:,:2]])
    [['N' 'CA']
     ['N' 'H']
     ['N' 'H2']
     ['CA' 'C']
     ['CA' 'CB']
     ['CA' 'HA']
     ['C' 'O']
     ['C' 'OXT']
     ['CB' 'HB1']
     ['CB' 'HB2']
     ['CB' 'HB3']
     ['OXT' 'HXT']]
    """
    _init_dataset()
    array_dict = _residues[res_name]

    array = AtomArray(len(array_dict["res_name"]))

    array.res_name = array_dict["res_name"]
    array.atom_name = array_dict["atom_name"]
    array.element = array_dict["element"]
    array.charge = array_dict["charge"]
    array.hetero = array_dict["hetero"]

    array.coord[:,0] = array_dict["coord_x"]
    array.coord[:,1] = array_dict["coord_y"]
    array.coord[:,2] = array_dict["coord_z"]

    array.bonds = BondList(
        array.array_length(),
        bonds = np.stack([
            array_dict["bond_i"],
            array_dict["bond_j"],
            array_dict["bond_type"]
        ]).T
    )

    return array