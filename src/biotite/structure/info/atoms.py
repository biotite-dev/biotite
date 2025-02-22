# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["residue"]

from biotite.structure.info.ccd import get_ccd

# fmt: off
NON_HETERO_RESIDUES = set([
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "PYL", "SER", "THR",
    "TRP", "TYR", "VAL", "SEC",
    "A", "DA", "G", "DG", "C", "DC", "U", "DT",
])
# fmt: on


def residue(res_name, allow_missing_coord=False):
    """
    Get an atom array, representing the residue with the given name.

    This atom array includes proper values for the ``residue_name``,
    ``hetero``, ``atom_name``, ``element`` and ``charge`` annotation
    arrays and bonds and coordinates.

    Parameters
    ----------
    res_name : str
        The up to 3-letter name of the residue.
    allow_missing_coord : bool, optional
        Whether to allow missing coordinate values in the residue.
        If ``True``, these will be represented as ``nan`` values.
        If ``False``, a ``ValueError`` is raised when missing coordinates
        are encountered.

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
    # Avoid circular import
    from biotite.structure.io.pdbx import get_component

    try:
        component = get_component(
            get_ccd(),
            res_name=res_name,
            allow_missing_coord=allow_missing_coord,
        )
    except KeyError:
        raise KeyError(f"No atom information found for residue '{res_name}' in CCD")
    component.hetero[:] = res_name not in NON_HETERO_RESIDUES
    return component
