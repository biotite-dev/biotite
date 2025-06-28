from os.path import join
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
import pytest
import biotite.interface.openmm as openmm_interface
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir


@pytest.fixture
def test_path():
    # Use a structure with hydrogen atoms
    return join(data_dir("structure"), "1l2y.cif")


@pytest.mark.parametrize("multi_state", [False, True])
def test_state_conversion(test_path, multi_state):
    """
    Test whether the :class:`AtomArray` obtained from a :class:`State`
    matches the original template atom array with newly generated
    positions.
    """
    pdbx_file = pdbx.CIFFile.read(test_path)
    template = pdbx.get_structure(pdbx_file, model=1)
    system = openmm_interface.to_system(template)
    # Create an arbitrary integrator
    integrator = openmm.VerletIntegrator(1)
    context = openmm.Context(system, integrator)

    # Generate arbitrary coordinates and box vectors
    coord = np.arange(template.array_length() * 3).reshape(-1, 3)
    box = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ]
    )
    context.setPositions(coord * unit.angstrom)
    context.setPeriodicBoxVectors(*(box * unit.angstrom))

    if multi_state:
        ref_atoms = struc.from_template(
            template, np.stack([coord] * 2), np.stack([box] * 2)
        )
    else:
        ref_atoms = template.copy()
        ref_atoms.coord = coord
        ref_atoms.box = box

    if multi_state:
        states = [context.getState(getPositions=True) for _ in range(2)]
        test_atoms = openmm_interface.from_states(template, states)
    else:
        state = context.getState(getPositions=True)
        test_atoms = openmm_interface.from_state(template, state)

    assert np.allclose(test_atoms.coord, ref_atoms.coord)
    assert np.allclose(test_atoms.box, ref_atoms.box)


def test_system_consistency(test_path):
    """
    Test whether a :class:`System` converted from a :class:`AtomArray` is equal to a
    :class:`System` directly read via OpenMM.
    Forces and constraints are not tested, as they are not set by :func:`to_system()`.
    """
    topology = app.PDBxFile(test_path).topology
    force_field = app.ForceField("amber14-all.xml")
    ref_system = force_field.createSystem(topology)

    atoms = pdbx.get_structure(pdbx.CIFFile.read(test_path), model=1)
    test_system = openmm_interface.to_system(atoms)

    assert test_system.getNumParticles() == ref_system.getNumParticles()
    test_masses = [
        test_system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(test_system.getNumParticles())
    ]
    ref_masses = [
        ref_system.getParticleMass(i).value_in_unit(unit.dalton)
        for i in range(ref_system.getNumParticles())
    ]
    assert test_masses == pytest.approx(ref_masses, abs=1e-2)
    assert (
        test_system.getDefaultPeriodicBoxVectors()
        == ref_system.getDefaultPeriodicBoxVectors()
    )


@pytest.mark.parametrize("include_box", [False, True])
def test_topology_conversion(test_path, include_box):
    """
    Converting an :class:`AtomArray` into a :class:`Topology` and back
    again should not change the :class:`AtomArray`.
    """
    ref_atoms = pdbx.get_structure(pdbx.CIFFile.read(test_path), model=1)
    ref_atoms.bonds = struc.connect_via_residue_names(ref_atoms)
    if not include_box:
        ref_atoms.box = None

    topology = openmm_interface.to_topology(ref_atoms)
    test_atoms = openmm_interface.from_topology(topology)

    # The Topology cannot properly handle the aromatic bond types of Biotite
    ref_atoms.bonds.remove_aromaticity()
    _assert_equal_atom_arrays(test_atoms, ref_atoms)


def test_topology_consistency(test_path):
    """
    Test whether an :class:`AtomArray` converted from a
    :class:`Topology` read via OpenMM is equal to :class:`AtomArray`
    directly read via Biotite.
    """
    ref_atoms = pdbx.get_structure(
        pdbx.CIFFile.read(test_path), model=1, extra_fields=["label_asym_id"]
    )
    # OpenMM uses author fields, except for the chain ID,
    # where it uses the label field
    ref_atoms.chain_id = ref_atoms.label_asym_id
    ref_atoms.del_annotation("label_asym_id")
    ref_atoms.bonds = struc.connect_via_residue_names(ref_atoms)

    topology = app.PDBxFile(test_path).topology
    test_atoms = openmm_interface.from_topology(topology)

    # OpenMM does not parse bond types for all bonds when parsing CIF files
    ref_atoms.bonds.remove_bond_order()
    test_atoms.bonds.remove_bond_order()
    # Biotite does not parse disulfide bridges
    # -> Remove them from the bonds parsed by OpenMM
    for i, j, _ in test_atoms.bonds.as_array():
        if test_atoms.element[i] == "S" and test_atoms.element[j] == "S":
            test_atoms.bonds.remove_bond(i, j)
    # The first residue has 'loose' hydrogen atoms which are automatically fixed by
    # OpenMM, resulting in a new name for the hydrogen atom and a new bond
    # For easy comparison, simply remove the first residue
    ref_atoms = ref_atoms[ref_atoms.res_id >= 2]
    test_atoms = test_atoms[test_atoms.res_id >= 2]

    _assert_equal_atom_arrays(test_atoms, ref_atoms)


def _assert_equal_atom_arrays(test_atoms, ref_atoms):
    for category in ref_atoms.get_annotation_categories():
        assert np.array_equal(
            test_atoms.get_annotation(category), ref_atoms.get_annotation(category)
        )

    if ref_atoms.box is not None:
        assert np.allclose(test_atoms.box, ref_atoms.box)
    else:
        assert test_atoms.box is None

    # Do not compare array from 'BondList.as_array()',
    # as the comparison would not allow different order of bonds
    assert test_atoms.bonds == ref_atoms.bonds
