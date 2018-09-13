# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides functions for calculation of mechanical or
mass-related properties of a molecular structure.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["atom_masses", "mass_of_element", "mass_center", "gyration_radius"]

import numpy as np
from .atoms import Atom, AtomArray, AtomArrayStack, coord
from .util import vector_dot, norm_vector
from .error import BadStructureError
from .geometry import distance


def gyration_radius(array, masses=None):
    """
    Compute the radius/radii of gyration of an atom array or stack.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array or stack to calculate the radius/radii of gyration
        for.
    masses : ndarray, optional
        The masses to use for each atom in the input `array`.
        Must have the same length as `array`. By default, the standard
        atomic mass for each element is taken.
    
    Returns
    -------
    masses : float or ndarray, dtype=float
        If `array` is an `AtomArray`, the radius of gyration is
        returned as single float.
        If `array` is an `AtomArrayStack`, an `ndarray` containing the
        radii of gyration for every model is returned.
    """
    if masses is None:
        masses = atom_masses(array)
    center = mass_center(array, masses)
    radii = distance(array, center[..., np.newaxis, :])
    inertia_moment = np.sum(masses * radii*radii, axis=-1)
    return np.sqrt(inertia_moment / np.sum(masses))

def mass_center(array, masses=None):
    """
    Calculate the center(s) of mass of an atom array or stack.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array or stack to calculate the center(s) of mass for.
    masses : ndarray, optional
        The masses to use for each atom in the input `array`.
        Must have the same length as `array`. By default, the standard
        atomic mass for each element is taken.
    
    Returns
    -------
    radius : ndarray, ndarray, dtype=float
        Array containing the the coordinates of the center of mass.
        If `array` is an `AtomArray`, this will be an length 3
        `ndarray`; if it is an `AtomArrayStack` with *n* models,
        a (*n x 3*) `ndarray` is returned.
    """
    if masses is None:
        masses = atom_masses(array)
    return np.sum(masses[:,np.newaxis] * array.coord, axis=-2) / np.sum(masses)


def atom_masses(array):
    """
    Obtain the atomic masses for an atom array. [1]_
    
    The weights are determined from the *element* annotation array.
    
    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array or stack to calculate the atom masses for.    
    
    Returns
    -------
    masses : np.ndarray, dtype=float
        Array containing the atom masses.
        The length is equal to the array length of the array or stack.
        
    References
    ----------
    
    .. [1] J Meija, TB Coplen, M Berglund, WA Brand, P De Bièvre,
       M Gröning, NE Holden, J Irrgeher, RD Loss, T Walczyk
       and T Prohaska
       "Atomic weights of the elements 2013 (IUPAC Technical Report)."
       Pure Appl Chem, 88, 265-291 (2016).
    
    """
    masses = np.zeros(array.array_length())
    for i, element in enumerate(array.element):
        try:
            masses[i] = _masses[element]
        except KeyError:
            raise ValueError(f"'{element}' is not a valid element")
    return masses


def mass_of_element(element):
    """
    Obtain the atomic mass of a given element. [1]_
    
    Parameters
    ----------
    element : str
        A chemical element.    
    
    Returns
    -------
    mass : float
        Atomic mass of the given element.
        
    References
    ----------
    
    .. [1] J Meija, TB Coplen, M Berglund, WA Brand, P De Bièvre,
       M Gröning, NE Holden, J Irrgeher, RD Loss, T Walczyk
       and T Prohaska
       "Atomic weights of the elements 2013 (IUPAC Technical Report)."
       Pure Appl Chem, 88, 265-291 (2016).
    
    """
    return _masses[element.upper()]

# Masses are taken from http://www.sbcs.qmul.ac.uk/iupac/AtWt/ (2018/03/01)
_masses = {
    "H"  :   1.008,
    "HE" :   4.002,
    "LI" :   6.940,
    "BE" :   9.012,
    "B"  :  10.810,
    "C"  :  12.011,
    "N"  :  14.007,
    "O"  :  15.999,
    "F"  :  18.998,
    "NE" :  20.180,
    "NA" :  22.989,
    "MG" :  24.305,
    "AL" :  26.981,
    "SI" :  28.085,
    "P"  :  30.973,
    "S"  :  32.060,
    "CL" :  35.450,
    "AR" :  39.948,
    "K"  :  39.098,
    "CA" :  40.078,
    "SC" :  44.955,
    "TI" :  47.867,
    "V"  :  50.941,
    "CR" :  51.996,
    "MN" :  54.938,
    "FE" :  55.845,
    "CO" :  58.933,
    "NI" :  58.693,
    "CU" :  63.546,
    "ZN" :  65.380,
    "GA" :  69.723,
    "GE" :  72.630,
    "AS" :  74.921,
    "SE" :  78.971,
    "BR" :  79.904,
    "KR" :  83.798,
    "RB" :  85.468,
    "SR" :  87.620,
    "Y"  :  88.905,
    "ZR" :  91.224,
    "NB" :  92.906,
    "MO" :  95.950,
    "TC" :  97.000,
    "RU" : 101.070,
    "RH" : 102.905,
    "PD" : 106.420,
    "AG" : 107.868,
    "CD" : 112.414,
    "IN" : 114.818,
    "SN" : 118.710,
    "SB" : 121.760,
    "TE" : 127.600,
    "I"  : 126.904,
    "XE" : 131.293,
    "CS" : 132.905,
    "BA" : 137.327,
    "LA" : 138.905,
    "CE" : 140.116,
    "PR" : 140.907,
    "ND" : 144.242,
    "PM" : 145.000,
    "SM" : 150.360,
    "EU" : 151.964,
    "GD" : 157.250,
    "TB" : 158.925,
    "DY" : 162.500,
    "HO" : 164.930,
    "ER" : 167.259,
    "TM" : 168.934,
    "YB" : 173.045,
    "LU" : 174.967,
    "HF" : 178.490,
    "TA" : 180.947,
    "W"  : 183.840,
    "RE" : 186.207,
    "OS" : 190.230,
    "IR" : 192.217,
    "PT" : 195.084,
    "AU" : 196.966,
    "HG" : 200.592,
    "TL" : 204.380,
    "PB" : 207.200,
    "BI" : 208.980,
    "PO" : 209.000,
    "AT" : 210.000,
    "RN" : 222.000,
    "FR" : 223.000,
    "RA" : 226.000,
    "AC" : 227.000,
    "TH" : 232.038,
    "PA" : 231.035,
    "U"  : 238.028,
    "NP" : 237.000,
    "PU" : 244.000,
    "AM" : 243.000,
    "CM" : 247.000,
    "BK" : 247.000,
    "CF" : 251.000,
    "ES" : 252.000,
    "FM" : 257.000,
    "MD" : 258.000,
    "NO" : 259.000,
    "LR" : 262.000,
    "RF" : 267.000,
    "DB" : 270.000,
    "SG" : 269.000,
    "BH" : 270.000,
    "HS" : 270.000,
    "MT" : 278.000,
    "DS" : 281.000,
    "RG" : 281.000,
    "CN" : 285.000,
    "NH" : 286.000,
    "FL" : 289.000,
    "MC" : 289.000,
    "LV" : 293.000,
    "TS" : 293.000,
    "OG" : 294.000
}