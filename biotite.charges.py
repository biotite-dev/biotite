# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
This module provides two functions; one for the computation of the
electronegativity of the individual atoms of a given AtomArrray, and one
for the computation of the partial charges of the individual atoms of a
given AtomArray according to the algorithm of Gasteiger-Marsili
"""

__name__ = "biotite.charges"
__author__ = "Jacob Marcel Anter"
__all__ = ["electronegativity", "partial_charges"]

import numpy as np
import structure as struc
import .structure.info as info
import warnings


# Creating dictionary to retrieve parameters for 
# electronegativity computation from
# First level of dictionary represents atom name
# Second level represents hybridisation state

en_parameters = {
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

# Defining function for computing electronegativity
def electronegativity(atom_array, charges = None):
    """
    Computes the electronegativity of the individual atoms of a given AtomArray
    depending on the individual atom's formal charge and its hybridization 
    state.

    Parameters
    ----------
    atom_array: Exclusively AtomArrays can be inserted in this function, not 
        AtomArrayStacks
        The AtomArray to get the electronegativity values from.
    charges: (NumPy) array, optional
        If an array is inserted, the respective charge values are used.
        If not, all charges are arbitrarily set to zero.

    Returns
    -------
    en_values: ndarray, dtype=float
        The electronegativity values of the individual atoms comprised in
        'atom_array'.
    
    Note
    ----
    A BondList must be added to the AtomArray as annotation category,
    i. e. it must be associated to the AtomArray inserted into the
    function as in the following example:

    atom_array.bonds = struc.connect_via_residue_names(atom_array)

    The annotation category's name must be "bonds" as well since this is the 
    name that is checked in order to verify the presence of a BondList.

    Otherwise, an error will be raised and electronegativity values
    won't be delivered.
    """
    # Setting the variable "has_key_error" initially to "False"
    has_key_error = False
    # Setting the variable "has_type_error" initially to "False"
    has_type_error = False
    # Checking whether BondList is associated to AtomArray
    if atom_array.bonds is None:
        raise AttributeError(
            "The input AtomArray doesn't possess an associated BondList."
        )
    # Preparing warning in case of KeyError
    list_of_unparametrized_atoms = []
    # Creating list to store electronegativity values in
    en_values = np.zeros(atom_array.shape[0])
    # For CPU time reasons, a nested list containing all binding partners of
    # a respective atom of the AtomArray is created
    # Information about binding partners is retrieved from this list instead of
    # determining binding partners in each iteration step anew
    bonds = [atom_array.bonds.get_bonds(i)[0] for i in range(atom_array.shape[0])]
    for index in range(atom_array.shape[0]):
        element = atom_array.element[index]
        atom_name = atom_array.atom_name[index]
        # Determining hybridization state through amount of binding partners
        amount_of_binding_partners = \
        len(bonds[index])
        try:
            a, b, c = \
            en_parameters[element][amount_of_binding_partners]
        # Exception handling in case atom name is not found in dictionary
        except KeyError:
            en_values[index] = np.nan
            list_of_unparametrized_atoms.append(atom_name)
            has_key_error = True
            # Stopping the current iteration and continuing
            # with the next one since the electronegativity is
            # arbitrarily set to NaN
            continue
        # Retrieving charge of respective atom from annotation
        try:
            charge = int(charges[index])
            # Computing electronegativity in case charges are given
            en_value = a + b*charge + c*(charge**2)
            en_values[index] = en_value
        # Exception handling in case no charge is available
        # In this case, the charge is arbitrarily set to zero
        except TypeError:
            has_type_error = True
            en_values[index] = a
    if has_key_error == True:
        # Using NumPy's 'unique' function to ensure that each atom only occurs
        # once in the list
        unique_list = np.unique(list_of_unparametrized_atoms)
        # Considering proper punctuation for the warning string
        warnings.warn(
            f"Parameters required for computation of "
            f"electronegativity aren't available for the following "
            f"atoms: {', '.join(unique_list)}. "
            f"Their electronegativity is given as NaN (Not a Number).", 
            UserWarning
        )
    if has_type_error == True:
        warnings.warn(
            f"Charge array wasn't given as optional argument. "
            f"Therefore, all atoms are assumed to be uncharged.",
            UserWarning
        )
    return en_values



# Defining function for computing partial charge
def partial_charges(atom_array, iteration_step_num = 6, charges = None):
    """
    Computes the partial charge of the individual atoms comprised in a given
    AtomArray depending on their electronegativity.

    Parameters
    ----------
    atom_array: Exclusively AtomArrays can be inserted in this function, not 
        AtomArrayStacks
        The AtomArray to get the partial charge values from.
    iteration_step_num: integer, optional
        The number of iteration steps is an optional argument and can be 
        chosen by the user depending on the desired precision of the result. 
        If no value is entered by the user, the default value '6' will be 
        as Gasteiger and Marsili described this amount of iteration steps as
        sufficient.
    
    Returns
    -------
    partial_charges: ndarray, dtype=float
        The partial charge values of the individual atoms comprised in
        'atom_array'.
    """
    # Creating list to store partial charge values in
    partial_charges = np.zeros(atom_array.shape[0])
    # Checking whether BondList is associated to AtomArray
    if atom_array.bonds is None:
        raise AttributeError(
            "The input AtomArray doesn't possess an associated BondList."
        )
    # Checking whether user entered charge array or not
    # If not, an array consisting of zero entries will be instantiated
    if charges == None:
        charges = np.zeros(atom_array.shape[0])
    current_charges = charges
    # For CPU time reasons, a nested list containing all binding partners of
    # a respective atom of the AtomArray is created
    # Information about binding partners is retrieved from this list instead of
    # determining binding partners in each iteration step anew
    bonds = [atom_array.bonds.get_bonds(i)[0] for i in range(atom_array.shape[0])]
    next_charges = np.copy(current_charges)
    # Performing iteration as described in scientific
    # paper of Gasteiger and Marsilsi
    # Amount of iteration steps can be chosen by user
    # If no value is entered, default value of 6 is used
    for counter in list(range(1, iteration_step_num + 1)):
        en_values = electronegativity(atom_array, current_charges)
        # Iterating through the AtomArray
        for i in range(atom_array.shape[0]):
            bonds_to_considered_atom = bonds[i]
            # For atoms whose name is not available in the dictionary, but
            # which aren't involved in covalent bonds, e. g. metal ions,
            # the partial charge is set to the ion charge
            if len(bonds_to_considered_atom) == 0:
                next_charges[i] = current_charges[i]
                continue
            considered_en_value = en_values[i]
            # For atoms that are not available in the dictionary,
            # but which are incorporated into molecules,
            # the partial charge is set to NaN (Not a Number)
            if np.isnan(considered_en_value) == True:
                next_charges[i] = np.nan
            else:
                for j in bonds_to_considered_atom:
                    # Only the charge transfer to the currently considered atom is
                    # computed in order to ensure independence of the result from the
                    # order of iteration through the given AtomArray.
                    # Independence of order is not given in case of simultaneous
                    # partial equilibration of electronegativity.
                    neighbour_en_value = en_values[j]
                    if np.isnan(neighbour_en_value):
                        continue
                    elif neighbour_en_value > considered_en_value:
                        divisor = considered_en_value
                    else:
                        divisor = neighbour_en_value
                    charge_transfer = ((neighbour_en_value - considered_en_value) / divisor) * 0.5**counter
                    next_charges[i] += charge_transfer
        current_charges = np.copy(next_charges)
    partial_charges = current_charges
    return partial_charges