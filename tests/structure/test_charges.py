# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
from biotite.structure.info import residue
from biotite.structure import Atom
from biotite.structure import array
from biotite.structure import BondList
import warnings

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
    all atoms comprised in the array 'elements' inserted into the
    function.

    By doing so, the function accesses the nested dictionary
    'EN_PARAMETERS'. The values originate from a publication of Johann
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
        in the 'elements' array.
    
    References
    ----------
    .. [1] J Gasteiger and M Marsili,
       "Iterative partial equalization of orbital electronegativity- a
       rapid access to atomic charges"
       Tetrahedron, 36, 3219 - 3288 (1980).
    """
    parameters = np.zeros((elements.shape[0], 3))
    has_key_error = False
    # Preparing warning in case of KeyError
    list_of_unparametrized_elements = []
    for i, element in enumerate(elements):
        try:
            a, b, c = \
            EN_PARAMETERS[element][amount_of_binding_partners[i]]
            parameters[i, 0] = a
            parameters[i, 1] = b
            parameters[i, 2] = c
        except KeyError:
            parameters[i, :] = np.nan
            list_of_unparametrized_elements.append(element)
            has_key_error = True
    if has_key_error:
        # Using NumPy's 'unique' function to ensure that each atom only
        # occurs once in the list
        unique_list = np.unique(list_of_unparametrized_elements)
        # Considering proper punctuation for the warning string
        warnings.warn(
            f"Parameters required for computation of "
            f"electronegativity aren't available for the following "
            f"atoms: {', '.join(unique_list)}. "
            f"Their electronegativity is given as NaN (Not a Number).", 
            UserWarning
        )
    return parameters


def partial_charges(atom_array, iteration_step_num=6, charges=None):
    """
    Compute the partial charge of the individual atoms comprised in a
    given AtomArray depending on their electronegativity.

    The algorithm implemented here is the so-called PEOE algorithm
    (partial equalization of orbital electronegativity) developed by
    Johann Gasteiger and Mario Marsili. [1]_

    Parameters
    ----------
    atom_array: AtomArray, shape=(n,)
        The AtomArray to get the partial charge values for. Exclusively
        AtomArrays can be inserted in this function, not
        AtomArrayStacks.
    iteration_step_num: integer, optional
        The number of iteration steps is an optional argument and can be 
        chosen by the user depending on the desired precision of the
        result. If no value is entered by the user, the default value
        '6' will be used as Gasteiger and Marsili described this amount
        of iteration steps as sufficient. [1]_
    charges: ndarray, dtype=int, optional
        The array comprising the formal charges of the atoms comprised
        in the inserted AtomArray ('atom_array'). Note that if this
        parameter is omitted, the formal charges of all atoms will be
        arbitrarily set to zero.
    
    Returns
    -------
    partial_charges: ndarray, dtype=float
        The partial charge values of the individual atoms comprised in
        'atom_array'.
    
    Notes
    -----
    A BondList must be associated to the AtomArray inserted into the
    function as in the following example:

    atom_array.bonds = struc.connect_via_residue_names(atom_array)

    The annotation category name must be "bonds" as well since this is
    the name that is checked in order to verify the presence of a
    BondList.

    Otherwise, an error will be raised and electronegativity values
    won't be delivered.

    This step can be omitted if the AtomArray is obtained by accessing
    the Chemical Component Dictionary by using the function
    'biotite.structure.info.residue' as AtomArrays obtained in this way
    are already associated to BondLists.

    For the electronegativity of positively charged hydrogen, the
    value of 20.02 eV is used.

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
    >>> print(partial_charges(fluoromethane, 1))
    [ 0.11473086 -0.17542017  0.02022977  0.02022977  0.02022977]
    >>> print(partial_charges(fluoromethane, 6))
    [ 0.07915367 -0.25264294  0.05782976  0.05782976  0.05782976]
    """
    amount_of_binding_partners = np.zeros(atom_array.shape[0])
    elements = atom_array.element
    if atom_array.bonds is None:
        raise AttributeError(
            f"The input AtomArray doesn't possess an associated "
            f"BondList."
        )
    if charges is None:
        charges = np.zeros(atom_array.shape[0])
        warnings.warn(
            f"Charge array wasn't given as optional argument. "
            f"Therefore, all atoms' formal charge is assumed "
            f"to be zero.",
            UserWarning
        )
    # For CPU time reasons, a nested list containing all binding
    # partners of a respective atom of the AtomArray is created
    bonds = \
    [atom_array.bonds.get_bonds(i)[0] for i in \
    range(atom_array.shape[0])]
    damping = 1.0
    for list_num in range(len(bonds)):
        amount_of_binding_partners[list_num] = \
        len(bonds[list_num])
    parameters = _get_parameters(elements, amount_of_binding_partners)
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
        ones_vector = np.transpose(np.atleast_2d(np.full(
        atom_array.shape[0], 1)))
        charge_array = np.concatenate((ones_vector, column_charges,
        sq_column_charges), axis = 1)
        en_values = np.sum(parameters * charge_array, axis=1)
        # Computing electronegativity values in case of positive charge
        # which enter as divisor the equation for charge transfer
        pos_en_values = np.sum(parameters, axis = 1)
        # Substituting values for hydrogen with the special value
        pos_en_values = np.array([20.02 if i == 12.85 else i 
        for i in pos_en_values])
        for i, j, _ in atom_array.bonds.as_array():
            # For atoms that are not available in the dictionary,
            # but which are incorporated into molecules,
            # the partial charge is set to NaN
            if np.isnan(en_values[[i, j]]).any():
                if np.isnan(en_values[i]):
                    charges[i] = np.nan
                # Determining for which atom exactly no parameters are
                # available is necessary since the other atom, for which
                # there indeed are parameters, could be involved in
                # multiple bonds.
                # Therefore, setting both charges to NaN would falsify
                # the result.
                else:
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


# First testing partial charge of carbon in the molecules given in table
# 3 of the publication
# Since some of the molecules are not available in the Chemical 
# Components Dictionary, the  respective AtomArrays are constructed via
# Biotite and the coordinates are arbitrarily set to the origin since
# the decisive information is the BondList

# Creating atoms to build molecules with
carbon = Atom([0, 0, 0], atom_name="C", element="C")

carbon_1 = Atom([0, 0, 0], atom_name="C1", element="C")

carbon_2 = Atom([0, 0, 0], atom_name="C2", element="C")

hydrogen = Atom([0, 0, 0], element ="H")

oxygen = Atom([0, 0, 0], element ="O")

nitrogen = Atom([0, 0, 0], element ="N")

fluorine = Atom([0, 0, 0], element ="F")

# Building molecules
methane = array([carbon_1, hydrogen, hydrogen, hydrogen, hydrogen])
methane_bonds = BondList(
    methane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
methane.bonds = methane_bonds


ethane = array(
    [carbon_1, carbon, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen]
)
ethane_bonds = BondList(
    ethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)
ethane.bonds = ethane_bonds


ethylene = array(
    [carbon_1, carbon, hydrogen, hydrogen, hydrogen, hydrogen]
)
ethylene_bonds = BondList(
    ethylene.array_length(),
    np.array([[0,1], [0,2], [0,3], [1,4], [1,5]])
)
ethylene.bonds = ethylene_bonds


acetylene = array(
    [carbon_1, carbon, hydrogen, hydrogen]
)
acetylene_bonds = BondList(
    acetylene.array_length(),
    np.array([[0,1], [0,2], [1,3]])
)
acetylene.bonds = acetylene_bonds


fluoromethane = array(
    [carbon_1, fluorine, hydrogen, hydrogen, hydrogen]
)
fluoromethane_bonds = BondList(
    fluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
fluoromethane.bonds = fluoromethane_bonds


difluoromethane = array(
    [carbon_1, fluorine, fluorine, hydrogen, hydrogen]
)
difluoromethane_bonds = BondList(
    difluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
difluoromethane.bonds = difluoromethane_bonds


trifluoromethane = array(
    [carbon_1, fluorine, fluorine, fluorine, hydrogen]
)
trifluoromethane_bonds = BondList(
    trifluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
trifluoromethane.bonds = trifluoromethane_bonds


tetrafluoromethane = array(
    [carbon_1, fluorine, fluorine, fluorine, fluorine]
)
tetrafluoromethane_bonds = BondList(
    tetrafluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
tetrafluoromethane.bonds = tetrafluoromethane_bonds


fluoroethane = array(
    [carbon_1, carbon_2, fluorine, hydrogen, hydrogen, hydrogen, 
    hydrogen, hydrogen]
)
fluoroethane_bonds = BondList(
    fluoroethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)
fluoroethane.bonds = fluoroethane_bonds


trifluoroethane = array(
    [carbon_1, carbon_2, fluorine, fluorine, fluorine, hydrogen,
    hydrogen, hydrogen]
)
trifluoroethane_bonds = BondList(
    trifluoroethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)
trifluoroethane.bonds = trifluoroethane_bonds


methanole = array(
    [carbon_1, oxygen, hydrogen, hydrogen, hydrogen, hydrogen]
)
methanole_bonds = BondList(
    methanole.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5]])
)
methanole.bonds = methanole_bonds


dimethyl_ether = array(
    [carbon_1, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen, hydrogen]
)
DME_bonds = BondList(
    dimethyl_ether.array_length(),
    np.array([[0,2], [1,2], [0,3], [0,4], [0,5], [1,6], [1,7], [1,8]])
)
dimethyl_ether.bonds = DME_bonds


formaldehyde = array(
    [carbon_1, oxygen, hydrogen, hydrogen]
)
formaldehyde_bonds = BondList(
    formaldehyde.array_length(),
    np.array([[0,1], [0,2], [0,3]])
)
formaldehyde.bonds = formaldehyde_bonds


acetaldehyde = array(
    [carbon_1, carbon_2, oxygen, hydrogen, hydrogen, hydrogen, hydrogen]
)
acetaldehyde_bonds = BondList(
    acetaldehyde.array_length(),
    np.array([[0,1], [1,2], [0,3], [0,4], [0,5], [1,6]])
)
acetaldehyde.bonds = acetaldehyde_bonds


acetone = array(
    [carbon_1, carbon_2, carbon, oxygen, hydrogen, hydrogen, hydrogen,
    hydrogen, hydrogen, hydrogen]
)
acetone_bonds = BondList(
    acetone.array_length(),
    np.array([[0,1], [1,2], [1,3], [0,4], [0,5], [0,6], [2,7], [2,8],
    [2,9]])
)
acetone.bonds = acetone_bonds


hydrogen_cyanide = array(
    [carbon_1, nitrogen, hydrogen]
)
HC_bonds = BondList(
    hydrogen_cyanide.array_length(),
    np.array([[0,1], [0,2]])
)
hydrogen_cyanide.bonds = HC_bonds


acetonitrile = array(
    [carbon_1, carbon_2, nitrogen, hydrogen, hydrogen, hydrogen]
)
ACN_bonds = BondList(
    acetonitrile.array_length(),
    np.array([[0,1], [1,2], [0,3], [0,4], [0,5]])
)
acetonitrile.bonds = ACN_bonds

# For this purpose, parametrization via pytest is performed
@pytest.mark.parametrize("molecule, expected_results", [
    (methane, (-0.078,)),
    (ethane, (-0.068,)),
    (ethylene, (-0.106,)),
    (acetylene, (-0.122,)),
    (fluoromethane, (0.079,)),
    (difluoromethane, (0.23,)),
    (trifluoromethane, (0.38,)),
    (tetrafluoromethane, (0.561,)),
    (fluoroethane, (0.087,)), #-0.037
    (trifluoroethane, (0.387,)), #-0.039
    (methanole, (0.033,)),
    (dimethyl_ether, (0.036,)),
    (formaldehyde, (0.115,)),
    (acetaldehyde, (-0.009,)), #0.123
    (acetone, (-0.006,)), #0.131
    (hydrogen_cyanide, (0.051,)),
    (acetonitrile, (0.023,)) #0.06
])

def test_partial_charges(molecule, expected_results):
    charges = partial_charges(molecule)
    assert charges[molecule.atom_name == "C1"].tolist() == \
    pytest.approx(expected_results, abs=1e-2)


# Now, as a second test, it is verified whether the sum of all partial
# charges equals the sum of all formal charges (in our case zero since
# we are exclusively dealing with uncharged molecules)
# Fluoromethane is taken as example
def test_total_charge_zero(fluoroethane):
    total_charge = np.sum(partial_charges(fluoroethane))
    assert total_charge == pytest.approx(0, abs=1e-15)