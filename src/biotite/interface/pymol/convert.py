__name__ = "biotite.interface.pymol"
__author__ = "Patrick Kunzmann"
__all__ = ["to_model", "from_model"]

import warnings
import numpy as np
from chempy import Atom, Bond
from chempy.models import Indexed as IndexedModel
import biotite.structure as struc
from biotite.interface import LossyConversionWarning
from biotite.sequence import ProteinSequence

# Bond orders in PyMOL (https://pymolwiki.org/index.php/Valence):
# 1: single bond, 2: double bond, 3:triple bond, 4:delocalized
_FORMAL_BOND_ORDER = {
    struc.BondType.SINGLE: 1,
    struc.BondType.DOUBLE: 2,
    struc.BondType.TRIPLE: 3,
    struc.BondType.AROMATIC_SINGLE: 1,
    struc.BondType.AROMATIC_DOUBLE: 2,
}
_DELOKALIZED_BOND_ORDER = {
    struc.BondType.SINGLE: 1,
    struc.BondType.DOUBLE: 2,
    struc.BondType.TRIPLE: 3,
    struc.BondType.AROMATIC_SINGLE: 4,
    struc.BondType.AROMATIC_DOUBLE: 4,
    struc.BondType.AROMATIC: 4,
}
_BIOTITE_BOND_TYPES = {
    1: struc.BondType.SINGLE,
    2: struc.BondType.DOUBLE,
    3: struc.BondType.TRIPLE,
    4: struc.BondType.AROMATIC,
}


def to_model(atom_array, delocalize_bonds=False):
    """
    Convert an :class:`AtomArray` into a :class:`chempy.models.Indexed`
    object.

    Parameters
    ----------
    atom_array : AtomArray
        The structure to be converted.
    delocalize_bonds : bool, optional
        If set to true, use *PyMOL*'s delocalized bond order for aromatic bonds.
        Otherwise, always use formal bond orders.

    Returns
    -------
    chempy_model : Indexed
        The converted structure.
    """
    if not np.isfinite(atom_array.coord).all():
        raise ValueError("PyMOL does not support infinite or NaN coordinates")

    model = IndexedModel()

    annot_cat = atom_array.get_annotation_categories()
    for i in range(atom_array.array_length()):
        atom = Atom()
        atom.segi = atom_array.chain_id[i]
        atom.chain = atom_array.chain_id[i]
        atom.resi_number = atom_array.res_id[i]
        atom.ins_code = atom_array.ins_code[i]
        res_name = atom_array.res_name[i]
        atom.resn = res_name
        if len(res_name) == 1:
            atom.resn_code = res_name
        else:
            try:
                atom.resn_code = ProteinSequence.convert_letter_3to1(res_name)
            except KeyError:
                atom.resn_code = "X"
        atom.hetatm = 1 if atom_array.hetero[i] else 0
        atom.name = atom_array.atom_name[i]
        atom.symbol = atom_array.element[i]
        if "b_factor" in annot_cat:
            atom.b = atom_array.b_factor[i]
        if "occupancy" in annot_cat:
            atom.q = atom_array.occupancy[i]
        if "charge" in annot_cat:
            atom.formal_charge = atom_array.charge[i]
        atom.coord = tuple(atom_array.coord[..., i, :])
        atom.index = i + 1
        model.add_atom(atom)

    bond_order_mapping = (
        _DELOKALIZED_BOND_ORDER if delocalize_bonds else _FORMAL_BOND_ORDER
    )
    unmappable_bond_types = []
    if atom_array.bonds is not None:
        for i, j, bond_type in atom_array.bonds.as_array():
            bond = Bond()
            order = bond_order_mapping.get(bond_type)
            if order is None:
                unmappable_bond_types.append(bond_type)
                order = 1
            bond.order = order
            bond.index = [i, j]
            model.add_bond(bond)
    else:
        warnings.warn("The given atom array (stack) has no associated bond information")
    if unmappable_bond_types:
        warnings.warn(
            "The following bond types could not be mapped to PyMOL: "
            + ", ".join([struc.BondType(bt).name for bt in set(unmappable_bond_types)]),
            LossyConversionWarning,
        )

    return model


def from_model(chempy_model, include_bonds=False):
    """
    Convert a :class:`chempy.models.Indexed`
    object into an :class:`AtomArray`.

    The returned :class:`AtomArray` contains the optional annotation
    categories ``b_factor``, ``occupancy``, ``charge`` and
    ``altloc_id``.
    No *altloc* ID filtering is performed.

    Parameters
    ----------
    chempy_model : Indexed
        The ``chempy`` model.
    include_bonds : bool, optional
        If set to true, an associated :class:`BondList` will be created
        for the returned atom array.

    Returns
    -------
    atom_array : AtomArray
        The converted structure.
    """
    atoms = chempy_model.atom

    bonds = chempy_model.bond

    atom_array = struc.AtomArray(len(atoms))

    # Add annotation arrays
    atom_array.chain_id = np.array([a.chain for a in atoms], dtype="U3")
    atom_array.res_id = np.array([a.resi_number for a in atoms], dtype=int)
    atom_array.ins_code = np.array([a.ins_code for a in atoms], dtype="U1")
    atom_array.res_name = np.array([a.resn for a in atoms], dtype="U3")
    atom_array.hetero = np.array([a.hetatm for a in atoms], dtype=bool)
    atom_array.atom_name = np.array([a.name for a in atoms], dtype="U6")
    atom_array.element = np.array([a.symbol for a in atoms], dtype="U2")

    atom_array.set_annotation(
        "b_factor",
        np.array([a.b if hasattr(a, "b") else 0 for a in atoms], dtype=float),
    )
    atom_array.set_annotation(
        "occupancy",
        np.array([a.q if hasattr(a, "q") else 1.0 for a in atoms], dtype=float),
    )
    atom_array.set_annotation(
        "charge",
        np.array(
            [a.formal_charge if hasattr(a, "formal_charge") else 0 for a in atoms],
            dtype=int,
        ),
    )
    atom_array.set_annotation(
        "altloc_id",
        np.array([a.alt if hasattr(a, "alt") else "" for a in atoms], dtype="U1"),
    )

    # Set coordinates
    atom_array.coord = np.array([a.coord for a in atoms], dtype=np.float32)

    # Add bonds
    if include_bonds:
        bond_array = np.array(
            [[b.index[0], b.index[1], _BIOTITE_BOND_TYPES[b.order]] for b in bonds],
            dtype=np.uint32,
        )
        atom_array.bonds = struc.BondList(len(atoms), bond_array)

    return atom_array
