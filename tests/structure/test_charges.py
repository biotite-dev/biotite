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

# Defining constant for the special case of the electronegativity of posotively
# charged hydrogen (value given in electronvolt, as all electronegativity 
# values)
EN_POS_HYDROGEN = 20.02

def _get_parameters(elements, amount_of_binding_partners):
    """
    Gathers the parameters required for electronegativity computation of all
    atoms comprised in the array 'elements' inserted into the function.

    By doing so, the function accesses the nested dictionary 'EN_PARAMETERS',
    whose first level represents the element name and whose second level
    represents the amount of binding partners.
    The values originate from a publication of Johann Gasteiger and Mario
    Marsili.
    By the mean of a 'for' loop, the function iterates through the array
    'elements' element-wise and adds the three parameters a, b, and c to an
    array initialized in the beginning of the function.

    Parameters
    ----------
    elements: ndarray, dtype=str
        The array comprising the elememts which to retrieve the parameters for.
    amount_of_binding_partners: ndarray, dtype=int
        The array containing information about the amount of binding partners
        of the respective atom/element.
    
    Returns
    -------
    parameters: NumPy array, dtype=float, shape=(n,3)
        The array containing all three parameters required for the computation
        of the electronegativities of all atoms comprised in the 'elements'
        array.
    """
    # Initializing a Numpy array to store the parameters in
    parameters = np.zeros((elements.shape[0], 3))
    # Setting the variable "has_key_error" initially to "False"
    has_key_error = False
    # Preparing warning in case of KeyError
    list_of_unparametrized_elements = []
    # Retrieving parameters from dictionary with the help of a for loop
    for i, element in enumerate(elements):
        try:
            a, b, c = \
            EN_PARAMETERS[element][amount_of_binding_partners[i]]
            # Overwriting initial values of array
            parameters[i, 0] = a
            parameters[i, 1] = b
            parameters[i, 2] = c
        # Exception handling in case atom name is not found in dictionary
        except KeyError:
            parameters[i, :] = np.nan
            list_of_unparametrized_elements.append(element)
            has_key_error = True
    if has_key_error:
        # Using NumPy's 'unique' function to ensure that each atom only occurs
        # once in the list
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

# Defining function for computing partial charge
def partial_charges(atom_array, iteration_step_num = 6, charges = None):
    """
    Computes the partial charge of the individual atoms comprised in a given
    AtomArray depending on their electronegativity.

    The function internally uses the private function '_get_parameters' in
    to gather all parameters required for the computation of the
    electronegativity values. This is done once in the whole algorithm.
    Electronegativity values are obtained by highly efficient multiplication of
    Numpy arrays, followed by row-wise summation. The same is performed under
    the assumption that all atoms possess a formal charge of +1 in order to
    obtain the respective electronegativity values which enter the equation of
    charge transfer as divisor. However, for hydrogen, the special value of
    20.02 eV is used due to its special properties!!! Electronegativity values
    are computed in the beginning if each iteration step as they depend on
    charge, which in turn alters in each iteration step.
    The last step of an iteration step consists of the bond-wise iteration
    through the BondList associated to the inserted AtomArray, the
    computation of the respective charge transfers and updating the charges.

    Parameters
    ----------
    atom_array: AtomArray, shape=(n,)
        The AtomArray to get the partial charge values for. Exclusively
        AtomArrays can be inserted in this function, not AtomArrayStacks.
    iteration_step_num: integer, optional
        The number of iteration steps is an optional argument and can be 
        chosen by the user depending on the desired precision of the result. 
        If no value is entered by the user, the default value '6' will be used
        as Gasteiger and Marsili described this amount of iteration steps as
        sufficient.
    charges: ndarray, dtype=int, optional
        The array comprising the formal charges of the atoms comprised in the
        inserted AtomArray ('atom_array'). Note that if this parameter is
        omitted, the formal charges of all atoms will be arbitrarily set to
        zero.
    
    Returns
    -------
    partial_charges: ndarray, dtype=float
        The partial charge values of the individual atoms comprised in
        'atom_array'.
    
    Notes
    -----
    A BondList must be added to the AtomArray as annotation category,
    i. e. it must be associated to the AtomArray inserted into the
    function as in the following example:

    atom_array.bonds = struc.connect_via_residue_names(atom_array)

    The annotation category name must be "bonds" as well since this is the name
    that is checked in order to verify the presence of a BondList.

    Otherwise, an error will be raised and electronegativity values
    won't be delivered.

    This step can be omitted if the AtomArray is obtained by accessing the
    Chemical Component Dictionary by using the function
    'biotite.structure.info.residue' as AtomArrays obtained in this way are
    already associated to BondLists.


    The number of iteration steps is an optional argument and can be 
    chosen by the user depending on the desired precision of the result. 
    If no value is entered by the user, the default value '6' will be 
    as Gasteiger and Marsili described this amount of iteration steps as
    sufficient.

    Examples
    --------
    The molecule fluoromethane is taken as example since detailed information
    is given about the charges of this molecule in each iteration step in the
    respective publication of Gasteiger and Marsili.

    >>> fluoromethane = residue("CF0")
    >>> print(partial_charges(fluoromethane, 1))
    [ 0.11473086 -0.17542017  0.02022977  0.02022977  0.02022977]
    >>> print(partial_charges(fluoromethane, 6))
    [ 0.07915367 -0.25264294  0.05782976  0.05782976  0.05782976]
    """
    # Creating array to store amount of binding partners of respective atom in
    amount_of_binding_partners = np.zeros(atom_array.shape[0])
    # Retrieving array containing element names from AtomArray
    elements = atom_array.element
    # Checking whether BondList is associated to AtomArray
    if atom_array.bonds is None:
        raise AttributeError(
            "The input AtomArray doesn't possess an associated BondList."
        )
    # Checking whether user entered charge array or not
    # If not, an array consisting of zero entries will be instantiated
    # The formal charges of all atoms are arbitrarily set zo zero
    # Additionally, the user is warned
    if charges is None:
        charges = np.zeros(atom_array.shape[0])
        warnings.warn(
            f"Charge array wasn't given as optional argument. "
            f"Therefore, all atoms' formal charge is assumed to be zero.",
            UserWarning
        )
    # For CPU time reasons, a nested list containing all binding partners of
    # a respective atom of the AtomArray is created
    # Information about binding partners is retrieved from this list instead of
    # determining binding partners in each iteration step anew
    bonds = \
    [atom_array.bonds.get_bonds(i)[0] for i in range(atom_array.shape[0])]
    # Setting damping factor initially to 1
    damping = 1.0
    # Filling the array 'amount of binding partners' by iterating through the 
    # list 'bonds'
    for list_num in range(len(bonds)):
        amount_of_binding_partners[list_num] = \
        len(bonds[list_num])
    # Applying '_get_parameters' function in order to gather parameters
    parameters = _get_parameters(elements, amount_of_binding_partners)
    # Performing iteration as described in scientific
    # paper of Gasteiger and Marsilsi
    # Amount of iteration steps can be chosen by user
    # If no value is entered, default value of 6 is used
    for _ in range(iteration_step_num):
        # In the beginning of each iteration step, the damping factor is 
        # multiplied by 0.5 in order to guarantee rapid convergence
        damping *= 0.5
        # 'charges' must be converted into a column vector by addibg a second
        # dimension followed by transponation
        column_charges = np.transpose(np.atleast_2d(charges))
        # The same is performed for the squared values of 'charges'
        sq_column_charges = np.transpose(np.atleast_2d(charges**2))
        # Last but not least, a column vector consisting of ones with the
        # same length as the charges vectors is created
        ones_vector = np.transpose(np.atleast_2d(np.full(
        atom_array.shape[0], 1)))
        # Now, the three column vectors are merged to one array with the shape
        # (atom_array.shape[0], 3)
        charge_array = np.concatenate((ones_vector, column_charges,
        sq_column_charges), axis = 1)
        # Electronegativity values are computed in each iteration step
        # 'en_values' is an one-dimensional array, shape= (n), where n 
        # describes the amount of atoms comprised in the AtomArray
        en_values = np.sum(parameters * charge_array, axis = 1)
        # Computing electronegativity values in case of positive charge which
        # enter as divisor the equation for charge transfer
        # For hydrogen, however, the special value of 20.02 eV described in the
        # paper is used!!!
        pos_en_values = np.sum(parameters, axis = 1)
        # Substituting values for the hydrogen with the special value via list
        # comprehension
        # At each position where the entry corresponds to the sum of the
        # parameters for hydrogen (12.85), the special value for Hydrogen
        # is inserted
        # This is valid since the value 12.85 doesn't occur elsewhere as the 
        # sum of the three parameters a, b and c
        pos_en_values = np.array([20.02 if i == 12.85 else i 
        for i in pos_en_values])
        # Iterating through the AtomArray
        for i, j, _ in atom_array.bonds.as_array():
            # For atoms that are not available in the dictionary,
            # but which are incorporated into molecules,
            # the partial charge is set to NaN (Not a Number)
            if np.isnan(en_values[[i, j]]).any():
                if np.isnan(en_values[i]):
                    charges[i] = np.nan
                # Determining for which atom exactly no parameters are
                # available is necessary since the other atom, for which there
                # indeed are parameters, could be involved in multiple bonds.
                # Therefore, setting both charges to NaN would falsify the 
                # result.
                else:
                    charges[j] = np.nan
            else:
                # Independence of order is not affected by simultaneous partial
                # equilibration of electronegativity.
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

# Now performing tests
# First testing partial charge of carbon in the molecules given in table 3 of
# the publication
# Since some of the molecules are not available in the Chemical Components
# Dictionary, the  respective AtomArrays are constructed via Biotite and the
# coordinates are arbitrarily set to the origin since the decisive information
# is the BondList

# Creating atoms to build molecules with
# Creating carbon
carbon = Atom([0, 0, 0], element = "C")

# Creating hydrogen
hydrogen = Atom([0, 0, 0], element = "H")

# Creating oxygen
oxygen = Atom([0, 0, 0], element = "O")

# Creating nitrogen
nitrogen = Atom([0, 0, 0], element = "N")

# Creating fluorine
fluorine = Atom([0, 0, 0], element = "F")

# Creating methane
methane = array([carbon, hydrogen, hydrogen, hydrogen, hydrogen])
# Creating BondList and asssociating it with the AtomArray
methane_bonds = BondList(
    methane.array_length(),
    np.array([[0,1], [0,1], [0,3], [0,4]])
)
methane.bonds = methane_bonds

# Creating ethane
ethane = array(
    [carbon, carbon, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
ethane_bonds = BondList(
    ethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)
ethane.bonds = ethane_bonds

# Creating ethylene
ethylene = array(
    [carbon, carbon, hydrogen, hydrogen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
ethylene_bonds = BondList(
    ethylene.array_length(),
    np.array([[0,1], [0,2], [0,3], [1,4], [1,5]])
)
ethylene.bonds = ethylene_bonds

# Creating acetylene
acetylene = array(
    [carbon, carbon, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
acetylene_bonds = BondList(
    acetylene.array_length(),
    np.array([[0,1], [0,2], [1,3]])
)
acetylene.bonds = acetylene_bonds

# Creating fluoromethane
fluoromethane = array(
    [carbon, fluorine, hydrogen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
fluoromethane_bonds = BondList(
    fluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
fluoromethane.bonds = fluoromethane_bonds

# Creating difluoromethane
difluoromethane = array(
    [carbon, fluorine, fluorine, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
difluoromethane_bonds = BondList(
    difluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
difluoromethane.bonds = difluoromethane_bonds

# Creating trifluoromethane
trifluoromethane = array(
    [carbon, fluorine, fluorine, fluorine, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
trifluoromethane_bonds = BondList(
    trifluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
trifluoromethane.bonds = trifluoromethane_bonds

# Creating tetrafluoromethane
tetrafluoromethane = array(
    [carbon, fluorine, fluorine, fluorine, fluorine]
)
# Creating BondList and asssociating it with the AtomArray
tetrafluoromethane_bonds = BondList(
    tetrafluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
tetrafluoromethane.bonds = tetrafluoromethane_bonds

# Creating fluoroethane
fluoroethane = array(
    [carbon, carbon, fluorine, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
fluoroethane_bonds = BondList(
    fluoroethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)
fluoroethane.bonds = fluoroethane_bonds

# Creating 1,1,1-trifluoroethane
trifluoroethane = array(
    [carbon, carbon, fluorine, fluorine, fluorine, hydrogen, hydrogen,
    hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
trifluoroethane_bonds = BondList(
    trifluoroethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)
trifluoroethane.bonds = trifluoroethane_bonds

# Creating methanole
methanole = array(
    [carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
methanole_bonds = BondList(
    methanole.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5]])
)
methanole.bonds = methanole_bonds

# Creating dimethyl ether
dimethyl_ether = array(
    [carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
DME_bonds = BondList(
    dimethyl_ether.array_length(),
    np.array([[0,2], [1,2], [0,3], [0,4], [0,5], [1,6], [1,7], [1,8]])
)
dimethyl_ether.bonds = DME_bonds

# Creating formaldehyde
formaldehyde = array(
    [carbon, oxygen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
formaldehyde_bonds = BondList(
    formaldehyde.array_length(),
    np.array([[0,1], [0,2], [0,3]])
)
formaldehyde.bonds = formaldehyde_bonds

# Creating acetaldehyde
acetaldehyde = array(
    [carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
acetaldehyde_bonds = BondList(
    acetaldehyde.array_length(),
    np.array([[0,1], [1,2], [0,3], [0,4], [0,5], [1,6]])
)
acetaldehyde.bonds = acetaldehyde_bonds

# Creating acetone
acetone = array(
    [carbon, carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen,
    hydrogen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
acetone_bonds = BondList(
    acetone.array_length(),
    np.array([[0,1], [1,2], [1,3], [0,4], [0,5], [0,6], [2,7], [2,8],
    [2,9]])
)
acetone.bonds = acetone_bonds

# Creating hydrogen cyanide
hydrogen_cyanide = array(
    [carbon, nitrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
HC_bonds = BondList(
    hydrogen_cyanide.array_length(),
    np.array([[0,1], [0,2]])
)
hydrogen_cyanide.bonds = HC_bonds

# Creating acetonitrile
acetonitrile = array(
    [carbon, carbon, nitrogen, hydrogen, hydrogen, hydrogen]
)
# Creating BondList and asssociating it with the AtomArray
ACN_bonds = BondList(
    acetonitrile.array_length(),
    np.array([[0,1], [1,2], [0,3], [0,4], [0,5]])
)
acetonitrile.bonds = ACN_bonds

# For this purpose, parametrization via pytest is performed
@pytest.mark.parametrize("carbon_1, expected_results", [
    ("methane", -0.078),
    ("ethane", -0.068),
    ("ethylene", -0.106),
    ("acetylene", -0.122),
    ("fluoromethane", 0.079),
    ("difluoromethane", 0.23),
    ("trifluoromethane", 0.38),
    ("tetrafluoromethane", 0.561),
    ("fluoroethane", 0.087),
    ("trifluoroethane", 0.387),
    ("methanole", 0.033),
    ("dimethyl_ether", 0.036),
    ("formaldehyde", 0.115),
    ("acetaldehyde", -0.009),
    ("acetone", -0.006),
    ("hydrogen_cyanide", 0.051),
    ("acetonitrile", 0.023)
])

# First testing carbon atoms in position 1
def test_carbon_in_pos_one(carbon_1, expected_results_1):
    assert partial_charges(carbon_1)[0] == expected_results_1

@pytest.mark.parametrize("carbon_2, expected_results", [
    ("fluoroethane", -0.037),
    ("trifluoroethane", -0.039),
    ("acetaldehyde", 0.123),
    ("acetone", 0.131),
    ("acetonitrile", 0.06)
])

# Then testing carbon atoms in position 2
def test_carbon_in_pos_two(carbon_2, expected_results_2):
    assert partial_charges(carbon_2)[0] == expected_results_2

# Now, as a second test, it is verified whether the sum of all partial charges
# equals the sum of all formal charges (in our case zero since we are
# exclusively dealing with uncharged molecules)