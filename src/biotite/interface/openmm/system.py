# `AtomArray`'s annotation arrays are typed as concrete numpy arrays in the
# stub, but at runtime accept Python lists too (the setter converts).
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

__name__ = "biotite.interface.openmm"
__author__ = "Patrick Kunzmann"
__all__ = ["to_system", "to_topology", "from_topology"]

import warnings
from typing import Any, cast
import numpy as np
import openmm.app as app
import openmm.unit as unit
from openmm import System
from openmm.app import Topology
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.bonds import BondList, BondType
from biotite.structure.chains import get_chain_starts
from biotite.structure.error import BadStructureError
from biotite.structure.filter import filter_amino_acids, filter_nucleotides
from biotite.structure.info.masses import mass
from biotite.structure.residues import get_residue_starts
from biotite.typing import XYZ, K, NDArray1, NDArray2

_BOND_TYPE_TO_ORDER = {
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.QUADRUPLE: 4,
    BondType.AROMATIC_SINGLE: 1,
    BondType.AROMATIC_DOUBLE: 2,
    BondType.AROMATIC_TRIPLE: 3,
}


def to_system(
    atoms: AtomArray[Any] | AtomArrayStack[Any, Any],
) -> System:
    """
    Create a :class:`openmm.openmm.System` from an :class:`AtomArray` or
    :class:`AtomArrayStack`.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure to be converted.
        The box vectors are set from the ``box`` attribute.
        If multiple models are given the box of the first model is selected.

    Returns
    -------
    system : System
        The created :class:`openmm.openmm.System`.
    """
    system = System()

    for element in atoms.element.tolist():
        system.addParticle(mass(element))

    if atoms.box is not None:
        if isinstance(atoms, AtomArrayStack):
            # If an `AtomArrayStack`, the first box is chosen
            box = atoms.box[0]
        else:
            box = atoms.box
        if not _check_box_requirements(box):
            raise BadStructureError(
                "Box does not fulfill OpenMM's requirements for periodic boxes"
            )
        # openmm's `Unit.__rmul__` produces a Quantity at runtime,
        # but that interaction isn't reflected in numpy's `ndarray.__mul__` stub
        # -> cast `box`
        system.setDefaultPeriodicBoxVectors(*(cast(Any, box) * unit.angstrom))

    return system


def to_topology(
    atoms: AtomArray[Any] | AtomArrayStack[Any, Any],
) -> Topology:
    """
    Create a :class:`openmm.app.topology.Topology` from an :class:`AtomArray` or
    :class:`AtomArrayStack`.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The structure to be converted.
        An associated :class:`BondList` is required.

    Returns
    -------
    topology : Topology
        The created :class:`openmm.app.topology.Topology`.
    """
    if "atom_id" in atoms.get_annotation_categories():
        atom_id = atoms.atom_id
    else:
        atom_id = np.arange(atoms.array_length()) + 1

    chain_starts = get_chain_starts(atoms)
    res_starts = get_residue_starts(atoms)

    # Lists of chain, residue and atom objects that will be filled later
    chain_list = []
    residue_list = []
    atom_list = []
    # Each atom's index in the chain and residue list
    chain_idx = _generate_idx(chain_starts, atoms.array_length())
    res_idx = _generate_idx(res_starts, atoms.array_length())

    topology = Topology()

    ## Add atoms
    for start_i in chain_starts:
        chain_list.append(topology.addChain(id=atoms.chain_id[start_i]))
    for start_i in res_starts:
        residue_list.append(
            topology.addResidue(
                name=atoms.res_name[start_i],
                chain=chain_list[chain_idx[start_i]],
                insertionCode=atoms.ins_code[start_i],
                id=str(atoms.res_id[start_i]),
            )
        )
    for i in np.arange(atoms.array_length()):
        atom_list.append(
            topology.addAtom(
                name=atoms.atom_name[i],
                element=app.Element.getBySymbol(atoms.element[i]),
                residue=residue_list[res_idx[i]],
                id=str(atom_id[i]),
            )
        )

    ## Add bonds
    if atoms.bonds is None:
        raise BadStructureError("Input structure misses an associated BondList")
    for atom_i, atom_j, bond_type in atoms.bonds.as_array():
        topology.addBond(
            atom_list[atom_i],
            atom_list[atom_j],
            order=_BOND_TYPE_TO_ORDER.get(bond_type),
        )

    ## Add box
    if atoms.box is not None:
        if isinstance(atoms, AtomArrayStack):
            # If an `AtomArrayStack`, the first box is chosen
            box = atoms.box[0]
        else:
            box = atoms.box
        if not _check_box_requirements(box):
            raise BadStructureError(
                "Box does not fulfill OpenMM's requirements for periodic boxes"
            )
        topology.setPeriodicBoxVectors(cast(Any, box) * unit.angstrom)

    return topology


def from_topology(topology: Topology) -> AtomArray[Any]:
    """
    Create a :class:`AtomArray` from a :class:`openmm.app.topology.Topology`.

    Parameters
    ----------
    topology : Topology
        The topology to be converted.

    Returns
    -------
    atoms : AtomArray
        The created :class:`AtomArray`.
        As the :class:`openmm.app.topology.Topology` does not contain atom
        coordinates, the values of the :class:`AtomArray` ``coord``
        are set to *NaN*.

    Notes
    -----
    This function is especially useful for obtaining an updated
    template, if the original topology was modified
    (e.g. via :class:`openmm.app.modeller.Modeller`).
    """
    atoms = AtomArray(topology.getNumAtoms())

    chain_ids = []
    res_ids = []
    ins_codes = []
    res_names = []
    atom_names = []
    elements = []
    atom_ids = []
    for chain in topology.chains():
        chain_id = chain.id
        for residue in chain.residues():
            res_name = residue.name
            res_id = int(residue.id)
            ins_code = residue.insertionCode
            for atom in residue.atoms():
                chain_ids.append(chain_id)
                res_ids.append(res_id)
                ins_codes.append(ins_code)
                res_names.append(res_name)
                atom_names.append(atom.name.upper())
                elements.append(atom.element.symbol.upper())
                atom_ids.append(str(atom.id))
    atoms.chain_id = chain_ids
    atoms.res_id = res_ids
    atoms.ins_code = ins_codes
    atoms.res_name = res_names
    atoms.atom_name = atom_names
    atoms.element = elements
    atoms.hetero = ~(filter_amino_acids(atoms) | filter_nucleotides(atoms))
    atom_ids = np.array(atom_ids, dtype=str)
    try:
        atom_ids = atom_ids.astype(int)
    except ValueError:
        warnings.warn("Could not convert atom IDs to integers, keeping them as strings")
        atom_ids = np.array(atom_ids, dtype=str)
    atoms.set_annotation("atom_id", atom_ids)

    bonds = []
    atom_to_index = {atom: i for i, atom in enumerate(topology.atoms())}
    for bond in topology.bonds():
        order = bond.order if bond.order is not None else BondType.ANY
        bonds.append([atom_to_index[bond.atom1], atom_to_index[bond.atom2], order])
    atoms.bonds = BondList(atoms.array_length(), np.array(bonds))

    box = topology.getPeriodicBoxVectors()
    if box is None:
        atoms.box = None
    else:
        atoms.box = np.asarray(box.value_in_unit(unit.angstrom))

    return atoms


def _generate_idx(
    starts: NDArray1[K, np.integer], length: int
) -> NDArray1[K, np.integer]:
    # An array that is 1, at start positions and 0 everywhere else
    start_counter = np.zeros(length, dtype=int)
    start_counter[starts] = 1
    # The first index should be zero -> the first start is not counted
    start_counter[0] = 0
    return np.cumsum(start_counter)


def _check_box_requirements(box: NDArray2[XYZ, XYZ, np.floating]) -> bool:
    """
    Return True, if the given box fulfills *OpenMM*'s requirements for
    boxes, else otherwise.

    The first vector must be on the x-axis
    and the second vector must be on the xy-plane.
    """
    return np.all(np.triu(box, k=1) == 0).item()
