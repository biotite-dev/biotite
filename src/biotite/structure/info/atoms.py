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

    # Residuue data is taken from
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
                0  ALA N      N         2.281   26.213   12.804
                0  ALA CA     C         1.169   26.942   13.411
                0  ALA C      C         1.539   28.344   13.874
                0  ALA O      O         2.709   28.647   14.114
                0  ALA CB     C         0.601   26.143   14.574
                0  ALA OXT    O         0.523   29.194   13.997
                0  ALA H      H         2.033   25.273   12.493
                0  ALA H2     H         3.080   26.184   13.436
                0  ALA HA     H         0.399   27.067   12.613
                0  ALA HB1    H        -0.247   26.699   15.037
                0  ALA HB2    H         0.308   25.110   14.270
                0  ALA HB3    H         1.384   25.876   15.321
                0  ALA HXT    H         0.753   30.069   14.286
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