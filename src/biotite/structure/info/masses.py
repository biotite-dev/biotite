# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

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
        [description]
    is_residue : bool, optional
        [description] (the default is None, which [default_description])
    
    Returns
    -------
    mass : float or None
        The mass of the given object. None if the mass is unknown.
    
    References
    ----------

    .. [1] J Meija, TB Coplen, M Berglund, WA Brand, P De Bièvre,
        M Gröning, NE Holden, J Irrgeher, RD Loss, T Walczyk
        and T Prohaska
        "Atomic weights of the elements 2013 (IUPAC Technical Report)."
        Pure Appl Chem, 88, 265-291 (2016).
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