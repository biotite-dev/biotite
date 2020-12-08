# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides one function for the computation of the partial
charges of the individual atoms of a given AtomArray according to the
PEOE algorithm of Gasteiger-Marsili.
"""

__name__ = "biotite.charges"
__author__ = "Jacob Marcel Anter"
__all__ = ["partial_charges"]

import numpy as np
from .info import residue
import warnings


# Creating dictionary to retrieve parameters for 
# electronegativity computation from
# First level of dictionary represents atom name
# Second level represents hybridisation state

EN_PARAMETERS = {
    "H": {
        1: (7.17, 6.24, -0.56)
    },

    "C": {
        4: (7.98, 9.18, 1.88),
        3: (8.79, 9.18, 1.88),
        2: (10.39, 9.45, 0.73)
    },

    "N": {
        # Considering protonated, e. g. in terminal
        # amino group (4 binding partners), as well
        # as unprotonated nitrogen (3 binding partners)
        4: (11.54, 10.82, 1.36),
        3: (11.54, 10.82, 1.36),
        2: (12.87, 11.15, 0.85),
        1: (15.68, 11.7, -0.27)
    },

    "O": {
        2: (14.18, 12.92, 1.39),
        1: (17.07, 13.79, 0.47)
    },

    "S": {
        2: (10.14, 9.13, 1.38)
    },

    "F": {
        1: (14.66, 13.85, 2.31)
    },

    "Cl": {
        1: (11.00, 9.69, 1.35)
    },

    "Br": {
        1: (10.08, 8.47, 1.16)
    },

    "I": {
        1: (9.90, 7.96, 0.96)
    }
}

# Defining constant for the special case of the electronegativity of
# positively charged hydrogen (value given in electronvolt, as all
# electronegativity values)
EN_POS_HYDROGEN = 20.02

def _get_parameters(elements, amount_of_binding_partners):
    """
    Gather the parameters required for electronegativity computation of
    all atoms comprised in the input array `elements`.

    By doing so, the function accesses the nested dictionary
    ``EN_PARAMETERS``. The values originate from a publication of Johann
    Gasteiger and Mario Marsili. [1]_

    Parameters
    ----------
    elements: ndarray, dtype=str
        The array comprising the elememts which to retrieve the
        parameters for.
    amount_of_binding_partners: ndarray, dtype=int
        The array containing information about the amount of binding
        partners of the respective atom/element.
    
    Returns
    -------
    parameters: NumPy array, dtype=float, shape=(n,3)
        The array containing all three parameters required for the
        computation of the electronegativities of all atoms comprised
        in the `elements` array.
    
    References
    ----------
    .. [1] J Gasteiger and M Marsili,
       "Iterative partial equalization of orbital electronegativity - a
       rapid access to atomic charges"
       Tetrahedron, 36, 3219 - 3288 (1980).
    """
    parameters = np.zeros((elements.shape[0], 3))
    has_atom_key_error = False
    has_valence_key_error = False
    # Preparing warning in case of KeyError
    # It is differentiated between atoms that are not parametrized at
    # all and specific valence states that are parametrized
    list_of_unparametrized_elements = []
    unparametrized_valences = []
    unparam_valence_names = []
    for i, element in enumerate(elements):
        try:
            a, b, c = \
                EN_PARAMETERS[element][amount_of_binding_partners[i]]
            parameters[i, 0] = a
            parameters[i, 1] = b
            parameters[i, 2] = c
        except KeyError:
            # Considering the special case of ions
            if amount_of_binding_partners[i] == 0:
                parameters[i, :] = np.nan
                continue
            try:
                EN_PARAMETERS[element]
            except KeyError:
                list_of_unparametrized_elements.append(element)
                has_atom_key_error = True
            else:
                unparam_valence_names.append(element)
                unparametrized_valences.append(
                    amount_of_binding_partners[i]
                )
                has_valence_key_error = True
            parameters[i, :] = np.nan
    if has_valence_key_error:
        joined_list = []
        for i in range(len(unparam_valence_names)):
            joined_list.append(
                unparam_valence_names[i]
                +
                " " * (10 - len(unparam_valence_names[i]))
            )
            joined_list.append(str(unparametrized_valences[i]) + "\n")
            joined_array = np.reshape(
                joined_list,
                newshape=(int(len(joined_list) / 2), 2)
            )
            joined_array = np.unique(joined_array, axis=0)
            # Array must be flattened in order ro be able to apply the
            # 'join' method
            flattened_joined_array = np.reshape(
                joined_array, newshape=(2*joined_array.shape[0])
            )
        warnings.warn(
            f"Parameters for specific valence states of some atoms "
            f"are not available. These valence states are: \n"
            f"Atom:     Amount of binding partners:\n"
            f"{''.join(flattened_joined_array)}"
            f"Their electronegativity is given as NaN.",
            UserWarning
        )
    if has_atom_key_error:
        # Using NumPy's 'unique' function to ensure that each atom only
        # occurs once in the list
        unique_list = np.unique(list_of_unparametrized_elements)
        # Considering proper punctuation for the warning string
        warnings.warn(
            f"Parameters required for computation of "
            f"electronegativity aren't available for the following "
            f"atoms: {', '.join(unique_list)}. "
            f"Their electronegativity is given as NaN.", 
            UserWarning
        )
    return parameters


def partial_charges(atom_array, iteration_step_num=6, charges=None):
    """
    Compute the partial charge of the individual atoms comprised in a
    given :class:`AtomArray` depending on their electronegativity.

    This function implements the
    *partial equalization of orbital electronegativity* (PEOE)
    algorithm [1]_.

    Parameters
    ----------
    atom_array: AtomArray, shape=(n,)
        The :class:`AtomArray` to get the partial charge values for.
        Must have an associated `BondList`.
    iteration_step_num: int, optional
        The number of iteration steps is an optional argument and can be 
        chosen by the user depending on the desired precision of the
        result. If no value is entered by the user, the default value
        '6' will be used.
        Gasteiger and Marsili described this number as sufficient [1]_.
    charges: ndarray, dtype=int, optional
        The array comprising the formal charges of the atoms in the
        input `atom_array`.
        If none is given, the ``charge`` annotation category of the
        input `atom_array` is used.
        If neither of them is given, the formal charges of all atoms
        will be arbitrarily set to zero.
    
    Returns
    -------
    partial_charges: ndarray, dtype=float
        The partial charge values of the individual atoms in the input
        `atom_array`.
    
    Notes
    -----
    A :class:`BondList` must be associated to the input
    :class:`AtomArray`.
    Otherwise, an error will be raised.
    Example:

    .. code-block:: python

        atom_array.bonds = struc.connect_via_residue_names(atom_array)

    |

    For the electronegativity of positively charged hydrogen, the
    value of 20.02 eV is used.

    Also note that the algorithm used in this function does not deliver
    proper results for expanded pi-electron systems like aromatic rings.

    References
    ----------
    .. [1] J Gasteiger and M Marsili,
       "Iterative partial equalization of orbital electronegativity- a
       rapid access to atomic charges"
       Tetrahedron, 36, 3219 - 3288 (1980).

    Examples
    --------
    The molecule fluoromethane is taken as example since detailed
    information is given about the charges of this molecule in each
    iteration step in the respective publication of Gasteiger and
    Marsili. [1]_

    >>> fluoromethane = residue("CF0")
    >>> print(fluoromethane.atom_name)
    ['C1' 'F1' 'H1' 'H2' 'H3']
    >>> print(partial_charges(fluoromethane, iteration_step_num=1))
    [ 0.115 -0.175  0.020  0.020  0.020]
    >>> print(partial_charges(fluoromethane, iteration_step_num=6))
    [ 0.079 -0.253  0.058  0.058  0.058]
    """
    amount_of_binding_partners = np.zeros(atom_array.shape[0])
    elements = atom_array.element
    if atom_array.bonds is None:
        raise AttributeError(
            f"The input AtomArray doesn't possess an associated "
            f"BondList."
        )
    if charges is None:
        try:
            # Implicitly this creates a copy of the charges
            charges = atom_array.charge.astype(np.float)
        except AttributeError:
            charges = np.zeros(atom_array.shape[0])
            warnings.warn(
                f"A charge array was neither given as optional "
                f"argument, nor does a charge annotation of the "
                f"inserted AtomArray exist. Therefore, all atoms' "
                f"formal charge is assumed to be zero.",
                UserWarning
            )

    bonds, _ = atom_array.bonds.get_all_bonds()
    amount_of_binding_partners = np.count_nonzero(bonds != -1, axis=1)
    damping = 1.0
    parameters = _get_parameters(elements, amount_of_binding_partners)
    # Computing electronegativity values in case of positive charge
    # which enter as divisor the equation for charge transfer
    pos_en_values = np.sum(parameters, axis=1)
    # Substituting values for hydrogen with the special value
    pos_en_values[atom_array.element == "H"] = EN_POS_HYDROGEN
    for _ in range(iteration_step_num):
        # In the beginning of each iteration step, the damping factor is 
        # halved in order to guarantee rapid convergence
        damping *= 0.5
        # For performing matrix-matrix-multiplication, the array
        # containing the charges, the array containing the squared
        # charges and another array consisting of entries of '1' and
        # having the same length as the previous two are converted into
        # column vectors and then merged to one array
        column_charges = np.transpose(np.atleast_2d(charges))
        sq_column_charges = np.transpose(np.atleast_2d(charges**2))
        ones_vector = np.transpose(
            np.atleast_2d(np.full(atom_array.shape[0], 1))
        )
        charge_array = np.concatenate(
            (ones_vector, column_charges,sq_column_charges), axis=1
        )
        en_values = np.sum(parameters * charge_array, axis=1)
        for i, j, _ in atom_array.bonds.as_array():
            # For atoms that are not available in the dictionary,
            # but which are incorporated into molecules,
            # the partial charge is set to NaN
            if np.isnan(en_values[[i, j]]).any():
                # Determining for which atom exactly no parameters are
                # available is necessary since the other atom, for which
                # there indeed are parameters, could be involved in
                # multiple bonds.
                # Therefore, setting both charges to NaN would falsify
                # the result.
                # The case that both atoms are not parametrized must be
                # considered as well.
                if np.isnan(en_values[i]):
                    charges[i] = np.nan
                if np.isnan(en_values[j]):
                    charges[j] = np.nan
            else:
                if en_values[j] > en_values[i]:
                    divisor = pos_en_values[i]
                else:
                    divisor = pos_en_values[j]
                charge_transfer = ((en_values[j] - en_values[i]) /
                    divisor) * damping
                charges[i] += charge_transfer
                charges[j] -= charge_transfer
    partial_charges = charges
    return partial_charges