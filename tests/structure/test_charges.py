# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import warnings
import numpy as np
import pytest
from biotite.structure import Atom, BondList, array, partial_charges

# Test the partial charge of carbon in the molecules given in table
# 3 of the Gasteiger-Marsili publication
# Since some of the molecules are not available in the Chemical
# Components Dictionary, the respective AtomArrays are constructed via
# Biotite and the coordinates are arbitrarily set to the origin since
# the relevant information is the BondList

# Creating atoms to build molecules with
carbon = Atom([0, 0, 0], element="C")
hydrogen = Atom([0, 0, 0], element="H")
oxygen = Atom([0, 0, 0], element="O")
nitrogen = Atom([0, 0, 0], element="N")
fluorine = Atom([0, 0, 0], element="F")
sulfur = Atom([0, 0, 0], element="S")


# Building molecules
methane = array([carbon, hydrogen, hydrogen, hydrogen, hydrogen])
methane.bonds = BondList(
    methane.array_length(), np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1]])
)
mol_length = methane.array_length()
methane.charge = np.array([0] * mol_length)


ethane = array(
    [carbon, carbon, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen]
)
ethane.bonds = BondList(
    ethane.array_length(),
    np.array(
        [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1]]
    ),
)
mol_length = ethane.array_length()
ethane.charge = np.array([0] * mol_length)


ethylene = array([carbon, carbon, hydrogen, hydrogen, hydrogen, hydrogen])
ethylene.bonds = BondList(
    ethylene.array_length(),
    np.array([[0, 1, 2], [0, 2, 1], [0, 3, 1], [1, 4, 1], [1, 5, 1]]),
)
mol_length = ethylene.array_length()
ethylene.charge = np.array([0] * mol_length)


acetylene = array([carbon, carbon, hydrogen, hydrogen])
acetylene.bonds = BondList(
    acetylene.array_length(), np.array([[0, 1, 3], [0, 2, 1], [1, 3, 1]])
)
mol_length = acetylene.array_length()
acetylene.charge = np.array([0] * mol_length)


fluoromethane = array([carbon, fluorine, hydrogen, hydrogen, hydrogen])
fluoromethane.bonds = BondList(
    fluoromethane.array_length(), np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1]])
)
mol_length = fluoromethane.array_length()
fluoromethane.charge = np.array([0] * mol_length)


difluoromethane = array([carbon, fluorine, fluorine, hydrogen, hydrogen])
difluoromethane.bonds = BondList(
    difluoromethane.array_length(),
    np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1]]),
)
mol_length = difluoromethane.array_length()
difluoromethane.charge = np.array([0] * mol_length)


trifluoromethane = array([carbon, fluorine, fluorine, fluorine, hydrogen])
trifluoromethane.bonds = BondList(
    trifluoromethane.array_length(),
    np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1]]),
)
mol_length = trifluoromethane.array_length()
trifluoromethane.charge = np.array([0] * mol_length)


tetrafluoromethane = array([carbon, fluorine, fluorine, fluorine, fluorine])
tetrafluoromethane.bonds = BondList(
    tetrafluoromethane.array_length(),
    np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1]]),
)
mol_length = tetrafluoromethane.array_length()
tetrafluoromethane.charge = np.array([0] * mol_length)


fluoroethane = array(
    [carbon, carbon, fluorine, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen]
)
fluoroethane.bonds = BondList(
    fluoroethane.array_length(),
    np.array(
        [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1]]
    ),
)
mol_length = fluoroethane.array_length()
fluoroethane.charge = np.array([0] * mol_length)


trifluoroethane = array(
    [carbon, carbon, fluorine, fluorine, fluorine, hydrogen, hydrogen, hydrogen]
)
trifluoroethane.bonds = BondList(
    trifluoroethane.array_length(),
    np.array(
        [[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1], [1, 6, 1], [1, 7, 1]]
    ),
)
mol_length = trifluoroethane.array_length()
trifluoroethane.charge = np.array([0] * mol_length)


methanole = array([carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen])
methanole.bonds = BondList(
    methanole.array_length(),
    np.array([[0, 1, 1], [0, 2, 1], [0, 3, 1], [0, 4, 1], [1, 5, 1]]),
)
mol_length = methanole.array_length()
methanole.charge = np.array([0] * mol_length)


dimethyl_ether = array(
    [carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen, hydrogen]
)
dimethyl_ether.bonds = BondList(
    dimethyl_ether.array_length(),
    np.array(
        [
            [0, 2, 1],
            [1, 2, 1],
            [0, 3, 1],
            [0, 4, 1],
            [0, 5, 1],
            [1, 6, 1],
            [1, 7, 1],
            [1, 8, 1],
        ]
    ),
)
mol_length = dimethyl_ether.array_length()
dimethyl_ether.charge = np.array([0] * mol_length)


formaldehyde = array([carbon, oxygen, hydrogen, hydrogen])
formaldehyde.bonds = BondList(
    formaldehyde.array_length(), np.array([[0, 1, 2], [0, 2, 1], [0, 3, 1]])
)
mol_length = formaldehyde.array_length()
formaldehyde.charge = np.array([0] * mol_length)


acetaldehyde = array([carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen, hydrogen])
acetaldehyde.bonds = BondList(
    acetaldehyde.array_length(),
    np.array([[0, 1, 1], [1, 2, 2], [0, 3, 1], [0, 4, 1], [0, 5, 1], [1, 6, 1]]),
)
mol_length = acetaldehyde.array_length()
acetaldehyde.charge = np.array([0] * mol_length)


acetone = array(
    [
        carbon,
        carbon,
        carbon,
        oxygen,
        hydrogen,
        hydrogen,
        hydrogen,
        hydrogen,
        hydrogen,
        hydrogen,
    ]
)
acetone.bonds = BondList(
    acetone.array_length(),
    np.array(
        [
            [0, 1, 1],
            [1, 2, 1],
            [1, 3, 2],
            [0, 4, 1],
            [0, 5, 1],
            [0, 6, 1],
            [2, 7, 1],
            [2, 8, 1],
            [2, 9, 1],
        ]
    ),
)
mol_length = acetone.array_length()
acetone.charge = np.array([0] * mol_length)


hydrogen_cyanide = array([carbon, nitrogen, hydrogen])
hydrogen_cyanide.bonds = BondList(
    hydrogen_cyanide.array_length(), np.array([[0, 1, 3], [0, 2, 1]])
)
mol_length = hydrogen_cyanide.array_length()
hydrogen_cyanide.charge = np.array([0] * mol_length)


acetonitrile = array([carbon, carbon, nitrogen, hydrogen, hydrogen, hydrogen])
acetonitrile.bonds = BondList(
    acetonitrile.array_length(),
    np.array([[0, 1, 1], [1, 2, 3], [0, 3, 1], [0, 4, 1], [0, 5, 1]]),
)
mol_length = acetonitrile.array_length()
acetonitrile.charge = np.array([0] * mol_length)


# For this purpose, parametrization via pytest is performed
@pytest.mark.parametrize(
    "molecule, expected_results",
    [
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
        (acetonitrile, (0.023, 0.06)),
    ],
)
def test_partial_charges(molecule, expected_results):
    """
    Test whether the partial charges of the carbon atoms comprised in
    the molecules given in table 3 of the publication computed in this
    implementation correspond to the values given in the publication
    within a certain tolerance range.
    """
    charges = partial_charges(molecule)
    assert charges[molecule.element == "C"].tolist() == pytest.approx(
        expected_results, abs=1e-2
    )


@pytest.mark.parametrize(
    "molecule",
    [
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
        acetonitrile,
    ],
)
def test_total_charge_zero(molecule):
    """
    In the case of the 17 molecules given in table 3, it is verified
    whether the sum of all partial charges equals the sum
    of all formal charges (in our case zero since we are exclusively
    dealing with uncharged molecules).
    """
    total_charge = np.sum(partial_charges(molecule))
    assert total_charge == pytest.approx(0, abs=1e-6)


def test_pos_formal_charge():
    """
    Test whether the partial charge of carbon in methane behaves as
    expected if it carries a formal charge of +1. To be more precise,
    it is expected to be smaller than 1 since this is the value which
    negative charge is addded to during iteration and also greater than
    the partial charge of carbon in methane carrying no formal charge.
    """
    pos_methane = methane.copy()
    pos_methane.charge = np.array([1, 0, 0, 0, 0])

    ref_carb_part_charge = partial_charges(methane, iteration_step_num=6)[0]
    pos_carb_part_charge = partial_charges(pos_methane, iteration_step_num=6)[0]
    assert pos_carb_part_charge < 1
    assert pos_carb_part_charge > ref_carb_part_charge


def test_valence_state_not_parametrized():
    """
    Test case in which parameters for a certain valence state of a
    generally parametrized atom are not available.
    In our case, it is sulfur having a double bond, i. e. only one
    binding partner.
    For this purpose, the molecule thioformaldehyde consisting of a
    central carbon bound to two hydrogen atoms via single bonds and to
    one sulfur atom via a double bond is created and tested.
    The expectations are the following: the sulfur's partial charge to
    be NaN and the carbons's partial charge to be smaller than that of
    the two hydrogens. Furthermore, the respective warning is expected
    to be raised.
    """
    with pytest.warns(
        UserWarning,
        match=(
            "Parameters for specific valence states of some atoms are not available"
        ),
    ):
        thioformaldehyde = array([carbon, sulfur, hydrogen, hydrogen])
        thioformaldehyde.bonds = BondList(
            thioformaldehyde.array_length(), np.array([[0, 1, 2], [0, 2, 1], [0, 3, 1]])
        )
        mol_length = thioformaldehyde.array_length()
        thioformaldehyde.charge = np.array([0] * mol_length)
        charges = partial_charges(thioformaldehyde)
        sulfur_part_charge = charges[1]
        carb_part_charge = charges[0]
        hyd_part_charge = charges[2]
    assert np.isnan(sulfur_part_charge)
    assert carb_part_charge < hyd_part_charge


def test_correct_output_ions():
    """
    Ions such as sodium or potassium are not parametrized. However,
    their formal charge is taken as partial charge since they are not
    involved in covalent bonding.
    Hence, it is expected that no warning is raised.
    The test is performed with a sodium ion.
    """
    sodium = Atom([0, 0, 0], element="NA")
    sodium_array = array([sodium])
    # Sodium possesses a formal charge of +1
    sodium_array.charge = np.array([1])
    # Sodium is not involved in covalent bonding
    sodium_array.bonds = BondList(sodium_array.array_length())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sodium_charge = partial_charges(sodium_array, iteration_step_num=1)[0]
    assert sodium_charge == 1


def test_correct_output_charged_aa():
    """
    Identification of hybridisation state is primarily performed via the
    bond type; if the bond type is BondType.ANY, it is resorted to the
    amount of binding partners of the respective atom, which can lead to
    an erroneous identification if the respective atom possesses a
    formal charge due to one additional binding partner or the loss of
    one binding partner. As an example, oxygen of an hydroxyl group has
    an formal charge of -1 if it loses the hydrogen, but still has the
    same hybridisation state. Identification via the amount of binding
    partners would be wrong.

    Therefore, the aim of this test is to verify that a molecule, in our
    case the amino acid glycine, with charges due to the addition or the
    loss of protons, delivers values for the partial charges of the
    atoms where addition/loss of the protons has taken place in an
    expected manner.
    A glycine AtomArray is constructed such that the carboxyl group is
    deprotonated and the amino group is protonated.

    Precisely, in the case of the negatively charged oxygen atom in the
    carboxyl group, a lower electronegativity value and therefore a
    lower partial charge is expected in case of correct identification
    of the hybridisation state via the bond type than in case of an
    erroneous identification via the amount of binding partners (cf.
    equation 2 for electronegativity and parameters listed in table 1 of
    the respective publication of Gasteiger and Marsili). The case of
    the positively charged nitrogen in the amino group, a correct
    identification of the hybridisation state via the amount of binding
    partners is possible as an amount of four binding partners is
    unambiguous, i. e. only represents the sp3 hybridisation. Hence,
    it is verified whether both ways of the identification of the
    hybridisation state deliver approximately the same result (small
    deviation arises from the difference in the parameters for the
    oxygen atom of the hydroxyl group in the carboxyl group that
    propagates to the amino group).
    Moreover, it is verified whether the respective UserWarning about
    unspecified bond types throughout the whole AtomArray is raised.
    """

    glycine_charge = np.array([+1, 0, 0, 0, -1, 0, 0, 0, 0, 0])

    glycine_with_btype = array(
        [
            nitrogen,
            carbon,
            carbon,
            oxygen,
            oxygen,
            hydrogen,
            hydrogen,
            hydrogen,
            hydrogen,
            hydrogen,
        ]
    )
    glycine_with_btype.charge = glycine_charge
    glycine_with_btype.bonds = BondList(
        glycine_with_btype.array_length(),
        np.array(
            [
                [0, 1, 1],
                [0, 5, 1],
                [0, 6, 1],
                [0, 7, 1],
                [1, 2, 1],
                [1, 8, 1],
                [1, 9, 1],
                [2, 3, 2],
                [2, 4, 1],
            ]
        ),
    )

    glycine_without_btype = glycine_with_btype.copy()
    glycine_without_btype.charge = glycine_charge
    glycine_without_btype.bonds = BondList(
        glycine_without_btype.array_length(),
        np.array(
            [
                [0, 1, 0],
                [0, 5, 0],
                [0, 6, 0],
                [0, 7, 0],
                [1, 2, 0],
                [1, 8, 0],
                [1, 9, 0],
                [2, 3, 0],
                [2, 4, 0],
            ]
        ),
    )

    part_charges_with_btype = partial_charges(glycine_with_btype)
    with pytest.warns(UserWarning, match="Each atom's bond type is 0"):
        part_charges_without_btype = partial_charges(glycine_without_btype)

    # Nitrogen of the amino group has the index 0
    nitr_charge_with_btype = part_charges_with_btype[0]
    nitr_charge_without_btype = part_charges_without_btype[0]
    assert nitr_charge_with_btype == pytest.approx(nitr_charge_without_btype, abs=5e-4)

    # Oxygen of the hydroxyl group in the carboxyl group has the index 2
    oxyg_charge_with_btype = part_charges_with_btype[2]
    oxyg_charge_without_btype = part_charges_without_btype[2]
    assert oxyg_charge_with_btype < oxyg_charge_without_btype
    # Assert that difference between the two values is significant
    difference_oxyg_charges = abs(oxyg_charge_with_btype - oxyg_charge_without_btype)
    assert difference_oxyg_charges > 3e-2
