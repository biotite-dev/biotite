# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.interface.rdkit"
__author__ = "Patrick Kunzmann"
__all__ = ["to_mol", "from_mol"]

import warnings
from collections import defaultdict
import numpy as np
from rdkit.Chem.rdchem import Atom, Conformer, EditableMol, KekulizeException, Mol
from rdkit.Chem.rdchem import BondType as RDKitBondType
from rdkit.Chem.rdmolops import AddHs, Kekulize, SanitizeFlags, SanitizeMol
from biotite.interface.version import requires_version
from biotite.interface.warning import LossyConversionWarning
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.bonds import BondList, BondType
from biotite.structure.error import BadStructureError

_KEKULIZED_TO_AROMATIC_BOND_TYPE = {
    BondType.SINGLE: BondType.AROMATIC_SINGLE,
    BondType.DOUBLE: BondType.AROMATIC_DOUBLE,
    BondType.TRIPLE: BondType.AROMATIC_TRIPLE,
}
_BIOTITE_TO_RDKIT_BOND_TYPE = {
    BondType.ANY: RDKitBondType.UNSPECIFIED,
    BondType.SINGLE: RDKitBondType.SINGLE,
    BondType.DOUBLE: RDKitBondType.DOUBLE,
    BondType.TRIPLE: RDKitBondType.TRIPLE,
    BondType.QUADRUPLE: RDKitBondType.QUADRUPLE,
    BondType.AROMATIC_SINGLE: RDKitBondType.AROMATIC,
    BondType.AROMATIC_DOUBLE: RDKitBondType.AROMATIC,
    BondType.AROMATIC_TRIPLE: RDKitBondType.AROMATIC,
    BondType.AROMATIC: RDKitBondType.AROMATIC,
    # Dative bonds may lead to a KekulizeException and may potentially be deprecated
    # in the future (https://github.com/rdkit/rdkit/discussions/6995)
    BondType.COORDINATION: RDKitBondType.SINGLE,
}
_RDKIT_TO_BIOTITE_BOND_TYPE = {
    RDKitBondType.UNSPECIFIED: BondType.ANY,
    RDKitBondType.SINGLE: BondType.SINGLE,
    RDKitBondType.DOUBLE: BondType.DOUBLE,
    RDKitBondType.TRIPLE: BondType.TRIPLE,
    RDKitBondType.QUADRUPLE: BondType.QUADRUPLE,
    RDKitBondType.DATIVE: BondType.COORDINATION,
}


# `Conformer.SetPositions()` was added in RDKit 2024.09.1
@requires_version("rdkit", ">=2024.09.1")
def to_mol(
    atoms, kekulize=False, use_dative_bonds=False, include_annotations=("atom_name",)
):
    """
    Convert an :class:`.AtomArray` or :class:`.AtomArrayStack` into a
    :class:`rdkit.Chem.rdchem.Mol`.

    Parameters
    ----------
    atoms : AtomArray or AtomArrayStack
        The molecule to be converted.
    kekulize : bool, optional
        If set to true, aromatic bonds are represented by single, double and triple
        bonds.
        By default, aromatic bond types are converted to
        :attr:`rdkit.rdchem.BondType.AROMATIC`.
    use_dative_bonds : bool, optional
        If set to true, :attr:`BondType.COORDINATION` bonds are translated to
        :attr:`rdkit.rdchem.BondType.DATIVE` bonds instead of
        :attr:`rdkit.rdchem.BondType.SINGLE` bonds.
        This may have the undesired side effect that a
        :class:`rdkit.Chem.rdchem.KekulizeException` is raised for some molecules, when
        the returned :class:`rdkit.Chem.rdchem.Mol` is kekulized.
    include_annotations : list of str, optional
        Names of annotation arrays in `atoms` that are added as atom-level property with
        the same name to the returned :class:`rdkit.Chem.rdchem.Mol`.
        These properties can be accessed with :meth:`rdkit.Chem.rdchem.Mol.GetProp()`.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        The *RDKit* molecule.
        If the input `atoms` is an :class:`AtomArrayStack`, all models are included
        as conformers with conformer IDs starting from ``0``.

    Examples
    --------

    >>> from rdkit.Chem import MolToSmiles
    >>> alanine_atom_array = residue("ALA")
    >>> mol = to_mol(alanine_atom_array)
    >>> print(MolToSmiles(mol))
    [H]OC(=O)C([H])(N([H])[H])C([H])([H])[H]

    By default, ``'atom_name'`` is stored as property of each atom.

    >>> for atom in mol.GetAtoms():
    ...     print(atom.GetProp("atom_name"))
    N
    CA
    C
    O
    CB
    OXT
    H
    H2
    HA
    HB1
    HB2
    HB3
    HXT
    """
    mol = EditableMol(Mol())

    has_charge_annot = "charge" in atoms.get_annotation_categories()
    for i in range(atoms.array_length()):
        rdkit_atom = Atom(atoms.element[i].capitalize())
        if has_charge_annot:
            rdkit_atom.SetFormalCharge(atoms.charge[i].item())
        for annot_name in include_annotations:
            rdkit_atom.SetProp(annot_name, atoms.get_annotation(annot_name)[i].item())
        mol.AddAtom(rdkit_atom)

    if atoms.bonds is None:
        raise BadStructureError("An AtomArray with associated BondList is required")
    bonds = atoms.bonds.as_array()
    if kekulize:
        bonds = bonds.copy()
        bonds.remove_aromaticity()
    for atom_i, atom_j, bond_type in atoms.bonds.as_array():
        if not use_dative_bonds and bond_type == BondType.COORDINATION:
            bond_type = BondType.SINGLE
        mol.AddBond(
            atom_i.item(), atom_j.item(), _BIOTITE_TO_RDKIT_BOND_TYPE[bond_type]
        )

    # Create a proper 'frozen' Mol object
    mol = mol.GetMol()
    coord = atoms.coord
    if coord.ndim == 2:
        # Handle AtomArray and AtomArrayStack consistently
        coord = coord[None, :, :]
    for model_coord in coord:
        conformer = Conformer(mol.GetNumAtoms())
        # RDKit silently expects the data to be in C-contiguous order
        # Otherwise the coordinates would be completely misassigned
        # (https://github.com/rdkit/rdkit/issues/8221)
        conformer.SetPositions(np.ascontiguousarray(model_coord, dtype=np.float64))
        conformer.Set3D(True)
        mol.AddConformer(conformer)

    return mol


@requires_version("rdkit", ">=2020")
def from_mol(mol, conformer_id=None, add_hydrogen=None):
    """
    Convert a :class:`rdkit.Chem.rdchem.Mol` into an :class:`.AtomArray` or
    :class:`.AtomArrayStack`.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to be converted.
    conformer_id : int, optional
        The conformer to be converted.
        By default, an :class:`AtomArrayStack` with all conformers is returned.
    add_hydrogen : bool, optional
        If set to true, explicit hydrogen atoms are always added.
        If set to false, explicit hydrogen atoms are never added.
        By default, explicit hydrogen atoms are only added, if hydrogen atoms are not
        already present.

    Returns
    -------
    atoms : AtomArray or AtomArrayStack
        The converted atoms.
        An :class:`AtomArrayStack` is only returned, if the `conformer_id` parameter
        is not set.

    Notes
    -----
    All atom-level properties of `mol`
    (obtainable with :meth:`rdkit.Chem.rdchem.Mol.GetProp()`) are added as string-type
    annotation array with the same name.
    ``element`` and ``charge`` are not inferred from properties but from the
    dedicated attributes in the :class:`rdkit.Chem.rdchem.Mol` object.

    Examples
    --------

    >>> from rdkit.Chem import MolFromSmiles
    >>> from rdkit.Chem.rdDistGeom import EmbedMolecule
    >>> from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
    >>> from rdkit.Chem.rdmolops import AddHs
    >>> mol = MolFromSmiles("C[C@@H](C(=O)O)N")
    >>> mol = AddHs(mol)
    >>> # Create a 3D conformer
    >>> conformer_id = EmbedMolecule(mol)
    >>> UFFOptimizeMolecule(mol)
    0
    >>> alanine_atom_array = from_mol(mol, conformer_id)
    >>> print(alanine_atom_array)
                0             C        -1.067    1.111   -0.079
                0             C        -0.366   -0.241   -0.217
                0             C         1.128   -0.082   -0.117
                0             O         1.654    0.353    0.943
                0             O         1.932   -0.413   -1.203
                0             N        -0.865   -1.173    0.796
                0             H        -0.715    1.807   -0.871
                0             H        -2.165    0.980   -0.191
                0             H        -0.862    1.562    0.916
                0             H        -0.613   -0.650   -1.221
                0             H         2.938   -0.311   -1.154
                0             H        -0.590   -0.837    1.749
                0             H        -0.408   -2.103    0.649
    """
    if add_hydrogen is None:
        add_hydrogen = not _has_explicit_hydrogen(mol)
    if add_hydrogen:
        SanitizeMol(mol, SanitizeFlags.SANITIZE_ADJUSTHS)
        mol = AddHs(mol)

    rdkit_atoms = mol.GetAtoms()
    if rdkit_atoms is None:
        raise BadStructureError("Could not obtains atoms from Mol")

    if conformer_id is None:
        conformers = [conf for conf in mol.GetConformers() if conf.Is3D()]
        atoms = AtomArrayStack(len(conformers), len(rdkit_atoms))
        for i, conformer in enumerate(conformers):
            atoms.coord[i] = np.array(conformer.GetPositions())
    else:
        conformer = mol.GetConformer(conformer_id)
        atoms = AtomArray(len(rdkit_atoms))
        atoms.coord = np.array(conformer.GetPositions())

    extra_annotations = defaultdict(
        # Use 'object' dtype first, as the maximum string length is unknown
        lambda: np.full(atoms.array_length(), "", dtype=object)
    )
    atoms.add_annotation("charge", int)
    for rdkit_atom in rdkit_atoms:
        annot_names = rdkit_atom.GetPropNames()
        for annot_name in annot_names:
            extra_annotations[annot_name][rdkit_atom.GetIdx()] = rdkit_atom.GetProp(
                annot_name
            )
        atoms.element[rdkit_atom.GetIdx()] = rdkit_atom.GetSymbol().upper()
        atoms.charge[rdkit_atom.GetIdx()] = rdkit_atom.GetFormalCharge()
    for annot_name, array in extra_annotations.items():
        atoms.set_annotation(annot_name, array.astype(str))

    rdkit_bonds = list(mol.GetBonds())
    is_aromatic = np.array(
        [bond.GetBondType() == RDKitBondType.AROMATIC for bond in rdkit_bonds]
    )
    if np.any(is_aromatic):
        # Determine the kekulized order of aromatic bonds
        # Copy as 'Kekulize()' modifies the molecule in-place
        mol = Mol(mol)
        try:
            Kekulize(mol)
        except KekulizeException:
            warnings.warn(
                "Kekulization failed, "
                "using 'BondType.ANY' instead for aromatic bonds instead",
                LossyConversionWarning,
            )
        rdkit_bonds = list(mol.GetBonds())
    bond_array = np.full((len(rdkit_bonds), 3), BondType.ANY, dtype=np.uint32)
    for i, bond in enumerate(rdkit_bonds):
        bond_type = _RDKIT_TO_BIOTITE_BOND_TYPE.get(bond.GetBondType())
        if bond_type is None:
            warnings.warn(
                f"Bond type '{bond.GetBondType().name}' cannot be mapped to Biotite, "
                "using 'BondType.ANY' instead",
                LossyConversionWarning,
            )
            bond_type = BondType.ANY
        if is_aromatic[i]:
            try:
                bond_type = _KEKULIZED_TO_AROMATIC_BOND_TYPE[bond_type]
            except KeyError:
                bond_type = BondType.AROMATIC
                warnings.warn(
                    "Kekulization returned invalid bond type, "
                    "using generic 'BondType.AROMATIC' instead",
                    LossyConversionWarning,
                )
        bond_array[i, 0] = bond.GetBeginAtomIdx()
        bond_array[i, 1] = bond.GetEndAtomIdx()
        bond_array[i, 2] = bond_type
    atoms.bonds = BondList(atoms.array_length(), bond_array)

    return atoms


def _has_explicit_hydrogen(mol):
    return mol.GetNumAtoms() > mol.GetNumHeavyAtoms()
