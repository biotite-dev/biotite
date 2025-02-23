# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.interface.rdkit"
__author__ = "Patrick Kunzmann"
__all__ = ["to_mol", "from_mol"]

import warnings
from collections import defaultdict
import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit.Chem import SanitizeFlags
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
    BondType.ANY: Chem.BondType.UNSPECIFIED,
    BondType.SINGLE: Chem.BondType.SINGLE,
    BondType.DOUBLE: Chem.BondType.DOUBLE,
    BondType.TRIPLE: Chem.BondType.TRIPLE,
    BondType.QUADRUPLE: Chem.BondType.QUADRUPLE,
    BondType.AROMATIC_SINGLE: Chem.BondType.AROMATIC,
    BondType.AROMATIC_DOUBLE: Chem.BondType.AROMATIC,
    BondType.AROMATIC_TRIPLE: Chem.BondType.AROMATIC,
    BondType.AROMATIC: Chem.BondType.AROMATIC,
    # Dative bonds may lead to a KekulizeException and may potentially be deprecated
    # in the future (https://github.com/rdkit/rdkit/discussions/6995)
    BondType.COORDINATION: Chem.BondType.SINGLE,
}
_RDKIT_TO_BIOTITE_BOND_TYPE = {
    Chem.BondType.UNSPECIFIED: BondType.ANY,
    Chem.BondType.SINGLE: BondType.SINGLE,
    Chem.BondType.DOUBLE: BondType.DOUBLE,
    Chem.BondType.TRIPLE: BondType.TRIPLE,
    Chem.BondType.QUADRUPLE: BondType.QUADRUPLE,
    Chem.BondType.DATIVE: BondType.COORDINATION,
}
_STANDARD_ANNOTATIONS = frozenset(
    {
        "chain_id",
        "res_id",
        "ins_code",
        "res_name",
        "hetero",
        "atom_name",
        "element",
        "charge",
        "b_factor",
        "occupancy",
        "label_alt_id",
    }
)


# `Conformer.SetPositions()` was added in RDKit 2024.09.1
@requires_version("rdkit", ">=2024.09.1")
def to_mol(
    atoms,
    kekulize=False,
    use_dative_bonds=False,
    include_extra_annotations=(),
    explicit_hydrogen=True,
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
    include_extra_annotations : list of str, optional
        Names of annotation arrays in `atoms` that are added as atom-level property with
        the same name to the returned :class:`rdkit.Chem.rdchem.Mol`.
        These properties can be accessed with :meth:`rdkit.Chem.rdchem.Mol.GetProp()`.
        Note that standard annotations (e.g. ``'chain_id', 'atom_name', 'res_name'``)
        are always included per default. These standard annotations can be accessed
        with :meth:`rdkit.Chem.rdchem.Atom.GetPDBResidueInfo()` for each atom in the
        returned :class:`rdkit.Chem.rdchem.Mol`.
    explicit_hydrogen : bool, optional
        If set to true, the conversion process expects that all hydrogen atoms are
        explicit, i.e. each each hydrogen atom must be part of the :class:`AtomArray`.
        If set to false, the conversion process treats all hydrogen atoms as implicit
        and all hydrogen atoms in the :class:`AtomArray` are removed.

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

    By default, ``'atom_name'`` is stored in RDKit's PDBResidueInfo grouping
    for each atom. We can access it manually as below

    >>> for atom in mol.GetAtoms():
    ...     print(atom.GetPDBResidueInfo().GetName())
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
    hydrogen_mask = atoms.element == "H"
    if explicit_hydrogen:
        if not hydrogen_mask.any():
            warnings.warn(
                "No hydrogen found in the input, although 'explicit_hydrogen' is 'True'"
            )
    else:
        atoms = atoms[..., ~hydrogen_mask]

    mol = Chem.EditableMol(Chem.Mol())

    has_annot = frozenset(atoms.get_annotation_categories())
    extra_annot = set(include_extra_annotations) - _STANDARD_ANNOTATIONS

    for i in range(atoms.array_length()):
        rdkit_atom = Chem.Atom(atoms.element[i].capitalize())
        if explicit_hydrogen:
            rdkit_atom.SetNoImplicit(True)
        if "charge" in has_annot:
            rdkit_atom.SetFormalCharge(atoms.charge[i].item())

        # add standard pdb annotations
        rdkit_atom_res_info = Chem.AtomPDBResidueInfo(
            atomName=atoms.atom_name[i].item(),
            residueName=atoms.res_name[i].item(),
            chainId=atoms.chain_id[i].item(),
            residueNumber=atoms.res_id[i].item(),
            isHeteroAtom=atoms.hetero[i].item(),
            insertionCode=atoms.ins_code[i].item(),
        )
        if "occupancy" in has_annot:
            rdkit_atom_res_info.SetOccupancy(atoms.occupancy[i].item())
        if "b_factor" in has_annot:
            rdkit_atom_res_info.SetTempFactor(atoms.b_factor[i].item())
        if "label_alt_id" in has_annot:
            rdkit_atom_res_info.SetAltLoc(atoms.label_alt_id[i].item())
        rdkit_atom.SetPDBResidueInfo(rdkit_atom_res_info)

        # add extra annotations
        for annot_name in extra_annot:
            rdkit_atom.SetProp(annot_name, atoms.get_annotation(annot_name)[i].item())

        # add atom to molecule
        mol.AddAtom(rdkit_atom)

    if atoms.bonds is None:
        raise BadStructureError("An AtomArray with associated BondList is required")
    if kekulize:
        bonds = atoms.bonds.copy()
        bonds.remove_aromaticity()
    else:
        bonds = atoms.bonds
    for atom_i, atom_j, bond_type in bonds.as_array():
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
        conformer = Chem.Conformer(mol.GetNumAtoms())
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
        Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ADJUSTHS)
        mol = Chem.AddHs(mol, addCoords=False, addResidueInfo=False)

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
    atoms.add_annotation("b_factor", float)
    atoms.add_annotation("occupancy", float)
    atoms.add_annotation("label_alt_id", str)

    for rdkit_atom in rdkit_atoms:
        _atom_idx = rdkit_atom.GetIdx()

        # ... add standard annotations
        element = rdkit_atom.GetSymbol().upper().strip()
        atoms.element[_atom_idx] = element
        atoms.charge[_atom_idx] = rdkit_atom.GetFormalCharge()

        # ... add PDB related annotations
        residue_info = rdkit_atom.GetPDBResidueInfo()
        if residue_info is None:
            # ... default values for atoms with missing residue information
            residue_info = Chem.AtomPDBResidueInfo(
                atomName="",
                occupancy=0.0,
                tempFactor=float("nan"),
                altLoc=".",
            )
            if element == "H":
                # ... attempt inferring residue information from nearest heavy atom
                #     in case of a hydrogen atom without explicit residue information
                nearest_heavy_atom = rdkit_atom.GetNeighbors()[0]
                nearest_heavy_atom_res_info = nearest_heavy_atom.GetPDBResidueInfo()
                if nearest_heavy_atom_res_info is not None:
                    residue_info.SetChainId(nearest_heavy_atom_res_info.GetChainId())
                    residue_info.SetResidueName(
                        nearest_heavy_atom_res_info.GetResidueName()
                    )
                    residue_info.SetResidueNumber(
                        nearest_heavy_atom_res_info.GetResidueNumber()
                    )
                    residue_info.SetInsertionCode(
                        nearest_heavy_atom_res_info.GetInsertionCode()
                    )
                    residue_info.SetIsHeteroAtom(
                        nearest_heavy_atom_res_info.GetIsHeteroAtom()
                    )
                    residue_info.SetAltLoc(nearest_heavy_atom_res_info.GetAltLoc())

        atoms.chain_id[_atom_idx] = residue_info.GetChainId()
        atoms.res_id[_atom_idx] = residue_info.GetResidueNumber()
        atoms.ins_code[_atom_idx] = residue_info.GetInsertionCode()
        atoms.res_name[_atom_idx] = residue_info.GetResidueName()
        atoms.label_alt_id[_atom_idx] = residue_info.GetAltLoc()
        atoms.hetero[_atom_idx] = residue_info.GetIsHeteroAtom()
        atoms.b_factor[_atom_idx] = residue_info.GetTempFactor()
        atoms.occupancy[_atom_idx] = residue_info.GetOccupancy()
        atoms.atom_name[_atom_idx] = residue_info.GetName().strip()

        # ... add extra annotations
        annot_names = rdkit_atom.GetPropNames()
        for annot_name in annot_names:
            extra_annotations[annot_name][_atom_idx] = rdkit_atom.GetProp(annot_name)

    for annot_name, array in extra_annotations.items():
        atoms.set_annotation(annot_name, array.astype(str))

    rdkit_bonds = list(mol.GetBonds())
    is_aromatic = np.array(
        [bond.GetBondType() == Chem.BondType.AROMATIC for bond in rdkit_bonds]
    )
    if np.any(is_aromatic):
        # Determine the kekulized order of aromatic bonds
        # Copy as 'Kekulize()' modifies the molecule in-place
        mol = Chem.Mol(mol)
        try:
            Chem.Kekulize(mol)
        except Chem.KekulizeException:
            warnings.warn(
                "Kekulization failed, "
                "using 'BondType.AROMATIC' instead for aromatic bonds instead",
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
