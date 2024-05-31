# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["residue"]

from .ccd import get_ccd


non_hetero_residues = set([
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS",
    "ILE","LEU","LYS","MET","PHE","PRO","PYL","SER","THR",
    "TRP","TYR","VAL", "SEC",
    "A", "DA", "G", "DG", "C", "DC", "U", "DT",
])


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
                0  ALA N      N        -0.970    0.490    1.500
                0  ALA CA     C         0.260    0.420    0.690
                0  ALA C      C        -0.090    0.020   -0.720
                0  ALA O      O        -1.060   -0.680   -0.920
                0  ALA CB     C         1.200   -0.620    1.300
                0  ALA OXT    O         0.660    0.440   -1.740
                0  ALA H      H        -1.380   -0.420    1.480
                0  ALA H2     H        -0.680    0.660    2.450
                0  ALA HA     H         0.750    1.390    0.680
                0  ALA HB1    H         1.460   -0.330    2.320
                0  ALA HB2    H         0.720   -1.590    1.310
                0  ALA HB3    H         2.110   -0.680    0.700
                0  ALA HXT    H         0.440    0.180   -2.650
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
    # Avoid circular import
    from ..io.pdbx import get_component

    try:
        component = get_component(get_ccd(), res_name=res_name)
    except KeyError:
        raise KeyError(
            f"No atom information found for residue '{res_name}' in CCD"
        )
    component.hetero[:] = res_name not in non_hetero_residues
    return component
