# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import pytest
import numpy as np
from biotite.structure.info import residue
from biotite.structure import Atom
from biotite.structure import array
from biotite.structure import BondList
from biotite.structure import charges


# Test the partial charge of carbon in the molecules given in table
# 3 of the Gasteiger-Marsili publication
# Since some of the molecules are not available in the Chemical 
# Components Dictionary, the respective AtomArrays are constructed via
# Biotite and the coordinates are arbitrarily set to the origin since
# the relevant information is the BondList

# Creating atoms to build molecules with
carbon = Atom([0, 0, 0], element="C")

hydrogen = Atom([0, 0, 0], element ="H")

oxygen = Atom([0, 0, 0], element ="O")

nitrogen = Atom([0, 0, 0], element ="N")

fluorine = Atom([0, 0, 0], element ="F")

sulfur = Atom([0, 0, 0], element="S")

# Building molecules
methane = array([carbon, hydrogen, hydrogen, hydrogen, hydrogen])
methane.bonds = BondList(
    methane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)


ethane = array(
    [carbon, carbon, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen]
)
ethane.bonds = BondList(
    ethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)


ethylene = array(
    [carbon, carbon, hydrogen, hydrogen, hydrogen, hydrogen]
)
ethylene.bonds = BondList(
    ethylene.array_length(),
    np.array([[0,1], [0,2], [0,3], [1,4], [1,5]])
)


acetylene = array(
    [carbon, carbon, hydrogen, hydrogen]
)
acetylene.bonds = BondList(
    acetylene.array_length(),
    np.array([[0,1], [0,2], [1,3]])
)


fluoromethane = array(
    [carbon, fluorine, hydrogen, hydrogen, hydrogen]
)
fluoromethane.bonds = BondList(
    fluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)


difluoromethane = array(
    [carbon, fluorine, fluorine, hydrogen, hydrogen]
)
difluoromethane.bonds = BondList(
    difluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)


trifluoromethane = array(
    [carbon, fluorine, fluorine, fluorine, hydrogen]
)
trifluoromethane.bonds = BondList(
    trifluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)


tetrafluoromethane = array(
    [carbon, fluorine, fluorine, fluorine, fluorine]
)
tetrafluoromethane.bonds = BondList(
    tetrafluoromethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)


fluoroethane = array(
    [carbon, carbon, fluorine, hydrogen, hydrogen, hydrogen, 
    hydrogen, hydrogen]
)
fluoroethane.bonds = BondList(
    fluoroethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)


trifluoroethane = array(
    [carbon, carbon, fluorine, fluorine, fluorine, hydrogen,
    hydrogen, hydrogen]
)
trifluoroethane.bonds = BondList(
    trifluoroethane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5], [1,6], [1,7]])
)


methanole = array(
    [carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen]
)
methanole.bonds = BondList(
    methanole.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4], [1,5]])
)


dimethyl_ether = array(
    [carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen,
    hydrogen, hydrogen]
)
dimethyl_ether.bonds = BondList(
    dimethyl_ether.array_length(),
    np.array([[0,2], [1,2], [0,3], [0,4], [0,5], [1,6], [1,7], [1,8]])
)


formaldehyde = array(
    [carbon, oxygen, hydrogen, hydrogen]
)
formaldehyde.bonds = BondList(
    formaldehyde.array_length(),
    np.array([[0,1], [0,2], [0,3]])
)


acetaldehyde = array(
    [carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen]
)
acetaldehyde.bonds = BondList(
    acetaldehyde.array_length(),
    np.array([[0,1], [1,2], [0,3], [0,4], [0,5], [1,6]])
)


acetone = array(
    [carbon, carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen,
    hydrogen, hydrogen, hydrogen]
)
acetone.bonds = BondList(
    acetone.array_length(),
    np.array([[0,1], [1,2], [1,3], [0,4], [0,5], [0,6], [2,7], [2,8],
    [2,9]])
)


hydrogen_cyanide = array(
    [carbon, nitrogen, hydrogen]
)
hydrogen_cyanide.bonds = BondList(
    hydrogen_cyanide.array_length(),
    np.array([[0,1], [0,2]])
)


acetonitrile = array(
    [carbon, carbon, nitrogen, hydrogen, hydrogen, hydrogen]
)
acetonitrile.bonds = BondList(
    acetonitrile.array_length(),
    np.array([[0,1], [1,2], [0,3], [0,4], [0,5]])
)

# For this purpose, parametrization via pytest is performed
@pytest.mark.parametrize("molecule, expected_results", [
    (methane, (-0.078,)),
    (ethane, (-0.068, -0.068)),
    (ethylene, (-0.106, -0.106)),
    (acetylene, (-0.122, -0.122)),
    (fluoromethane, (0.079,)),
    (difluoromethane, (0.23,)),
    (trifluoromethane, (0.38,)),
    (tetrafluoromethane, (0.561,)),
    (fluoroethane, (0.087, -0.037)),
    (trifluoroethane, (0.387, 0.039)),
    (methanole, (0.033,)),
    (dimethyl_ether, (0.036, 0.036)),
    (formaldehyde, (0.115,)),
    (acetaldehyde, (-0.009, 0.123)),
    (acetone, (-0.006, 0.131, -0.006)),
    (hydrogen_cyanide, (0.051,)),
    (acetonitrile, (0.023, 0.06))
])
def test_partial_charges(molecule, expected_results):
    """
    Test whether the partial charges of the carbon atoms comprised in
    the molecules given in table 3 of the publication computed in this
    implementation correspond to the values given in the publication
    within a certain tolerance range.
    """
    charges_ = charges.partial_charges(molecule)
    assert charges_[molecule.element == "C"].tolist() == \
        pytest.approx(expected_results, abs=1e-2)


@pytest.mark.parametrize("molecule", [
    methane,
    ethane,
    ethylene,
    acetylene,
    fluoromethane,
    difluoromethane,
    trifluoromethane,
    tetrafluoromethane,
    fluoroethane,
    trifluoroethane,
    methanole,
    dimethyl_ether,
    formaldehyde,
    acetaldehyde,
    acetone,
    hydrogen_cyanide,
    acetonitrile
])
def test_total_charge_zero(molecule):
    """
    In the case of the 17 molecules given in table 3, it is verified
    whether the sum of all partial charges equals the sum
    of all formal charges (in our case zero since we are exclusively
    dealing with uncharged molecules).
    """
    total_charge = np.sum(charges.partial_charges(molecule))
    assert total_charge == pytest.approx(0, abs=1e-15)


# Test whether a formal charge of +1 has the expected effects on
# carbons's partial charge in methane
# AtomArray with positive carbon must be created ab inition in order to
# prevent influencing 'methane'
pos_methane = array(
    [carbon, hydrogen, hydrogen, hydrogen, hydrogen]
)
pos_methane.bonds = BondList(
    pos_methane.array_length(),
    np.array([[0,1], [0,2], [0,3], [0,4]])
)
pos_methane.charge = np.array([1, 0, 0, 0, 0])

def test_pos_formal_charge():
    """
    Test whether the partial charge of carbon in methane behaves as
    expected if it carries a formal charge of +1. To be more precise,
    it is expected to be smaller than 1 since this is the value which
    negative charge is addded to during iteration and also greater than
    the partial charge of carbon in methane carrying no formal charge.
    """
    ref_carb_part_charge = charges.partial_charges(
        methane, iteration_step_num=6
    )[0]
    pos_carb_part_charge = charges.partial_charges(
        pos_methane, iteration_step_num=6
    )[0]
    assert pos_carb_part_charge < 1
    assert pos_carb_part_charge > ref_carb_part_charge


def test_valence_state_not_parametrized():
    """
    Test case in which parameters for a certain valence state of a
    generally parametrized atom are not available.
    In our case, it is sulfur having a double bond, i. e. only one
    binding partner.
    For this purpose, a fictitious molecule consisting of a central
    carbon bound to two hydrogen atoms via single bonds and to one
    sulfur atom via a double bond is created and tested.
    The expectations are the following: the sulfur's partial charge to
    be NaN and the carbons's partial charge to be smaller than that of
    the two hydrogens.
    """
    fictitious_molecule = array(
        [carbon, sulfur, hydrogen, hydrogen]
    )
    fictitious_molecule.bonds = BondList(
        fictitious_molecule.array_length(),
        np.array([[0,1], [0,2], [0,3]])
    )
    sulfur_part_charge = charges.partial_charges(fictitious_molecule)[1]
    carb_part_charge = charges.partial_charges(fictitious_molecule)[0]
    hyd_part_charge = charges.partial_charges(fictitious_molecule)[2]
    assert np.isnan(sulfur_part_charge)
    assert carb_part_charge < hyd_part_charge