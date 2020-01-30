# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.structure.info"
__author__ = "Patrick Kunzmann"
__all__ = ["mass"]

import json
from os.path import join, dirname, realpath
import msgpack
from ..atoms import Atom, AtomArray, AtomArrayStack


_info_dir = dirname(realpath(__file__))
# Masses are taken from http://www.sbcs.qmul.ac.uk/iupac/AtWt/ (2018/03/01)
with open(join(_info_dir, "atom_masses.json"), "r") as file:
    _atom_masses = json.load(file)
# Masses are taken from
# ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
# (2019/01/27)
with open(join(_info_dir, "residue_masses.msgpack"), "rb") as file:
    _res_masses = msgpack.load(file, raw=False)


def mass(item, is_residue=None):
    """
    Calculate the mass for the given object. [1]_

    If a residue name is given, the mass values refer to the masses of
    the complete molecule without additional or missing protons.
    In case of residues in a longer chain, some atoms might be missing
    from the molecule.
    For example non-terminal residues in a protein or nucleotide chain
    miss the mass of a water molecule.
    
    Parameters
    ----------
    item : str or Atom or AtomArray or AtomArrayStack
        The atom or molecule to get the mass for.
        If a string is given, it is interpreted as residue name or
        chemical element.
        If an :class:`Atom` is given the mass is taken from its element.
        If an :class:`AtomArray` or :class:`AtomArrayStack` is given the
        mass is the sum of the mass of its atoms.
    is_residue : bool, optional
        If set to true and a string is given for `item`, the string
        will be strictly interpreted as residue.
        If set to false, the string is strictly interpreted as element.
        By default the string will be interpreted as element at first
        and secondly as residue name, if the element is unknown.
    
    Returns
    -------
    mass : float or None
        The mass of the given object in *u*. None if the mass is unknown.
    
    References
    ----------

    .. [1] J Meija, TB Coplen, M Berglund, WA Brand, P De Bièvre,
        M Gröning, NE Holden, J Irrgeher, RD Loss, T Walczyk
        and T Prohaska
        "Atomic weights of the elements 2013 (IUPAC Technical Report)."
        Pure Appl Chem, 88, 265-291 (2016).
    
    Examples
    --------

    >>> print(mass(atom_array))
    2170.438
    >>> first_residue = list(residue_iter(atom_array))[0]
    >>> print(first_residue)
        A       1  ASN N      N        -8.901    4.127   -0.555
        A       1  ASN CA     C        -8.608    3.135   -1.618
        A       1  ASN C      C        -7.117    2.964   -1.897
        A       1  ASN O      O        -6.634    1.849   -1.758
        A       1  ASN CB     C        -9.437    3.396   -2.889
        A       1  ASN CG     C       -10.915    3.130   -2.611
        A       1  ASN OD1    O       -11.269    2.700   -1.524
        A       1  ASN ND2    N       -11.806    3.406   -3.543
        A       1  ASN H1     H        -8.330    3.957    0.261
        A       1  ASN H2     H        -8.740    5.068   -0.889
        A       1  ASN H3     H        -9.877    4.041   -0.293
        A       1  ASN HA     H        -8.930    2.162   -1.239
        A       1  ASN HB2    H        -9.310    4.417   -3.193
        A       1  ASN HB3    H        -9.108    2.719   -3.679
        A       1  ASN HD21   H       -11.572    3.791   -4.444
        A       1  ASN HD22   H       -12.757    3.183   -3.294
    >>> print(mass("ASN"))
    132.118
    >>> first_atom = first_residue[0]
    >>> print(first_atom)
        A       1  ASN N      N        -8.901    4.127   -0.555
    >>> print(mass(first_atom))
    14.007
    >>> print(mass("N"))
    14.007
    """

    if isinstance(item, str):
        if is_residue is None:
            result_mass = _atom_masses.get(item.upper())
            if result_mass is None:
                result_mass = _res_masses.get(item.upper())
        elif not is_residue:
            result_mass = _atom_masses.get(item.upper())
        else:
            result_mass = _res_masses.get(item.upper())
    
    elif isinstance(item, Atom):
        result_mass = mass(item.element, is_residue=False)
    elif isinstance(item, AtomArray) or isinstance(item, AtomArrayStack):
        result_mass = sum(
            (mass(element, is_residue=False) for element in item.element)
        )
    
    else:
        raise TypeError(
            f"Cannot calculate mass for {type(item).__name__} objects"
        )
    
    if result_mass is None:
        raise KeyError(f"{item} is not known")
    return result_mass