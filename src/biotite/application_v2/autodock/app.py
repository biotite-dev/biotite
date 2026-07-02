# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.autodock"
__author__ = "Patrick Kunzmann"
__all__ = ["VinaApp", "VinaResult"]

from dataclasses import dataclass
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
from biotite.application_v2.localapp import (
    CLIOption,
    CLIParameter,
    CommandSetup,
    LocalApp,
    cleanup_tempfile,
    command,
)
from biotite.structure.atoms import AtomArray, AtomArrayStack
from biotite.structure.connect import find_connected
from biotite.structure.error import BadStructureError
from biotite.structure.io.pdbqt import PDBQTFile
from biotite.structure.residues import get_residue_masks, get_residue_starts_for


@dataclass(frozen=True)
class VinaResult:
    """
    The result of an *AutoDock Vina* docking run.

    Attributes
    ----------
    ligand_models : AtomArrayStack
        The docked ligand.
        Each model corresponds to one binding mode.
        The models are sorted from best to worst predicted binding
        affinity.
        The returned structure may contain less atoms than the input
        structure, as *Vina* removes nonpolar hydrogen atoms.
        Furthermore, the returned structure contains *AutoDock* atom
        types as ``element`` annotation.
    ligand_coord : ndarray, shape=(m,n,3), dtype=float
        The coordinates for *m* binding modes and *n* atoms
        of the input ligand.
        The models are sorted from best to worst predicted binding
        affinity.
        Missing coordinates due to the removed nonpolar hydrogen
        atoms are set to *NaN*.
    flexible_residue_models : AtomArrayStack
        The docked side chains.
        Each model corresponds to one binding mode.
        The models are sorted from best to worst predicted binding
        affinity.
        If no flexible side chains were defined, this
        :class:`AtomArrayStack` contains no atoms.
        The returned structure may contain less atoms than the input
        structure, as *Vina* removes nonpolar hydrogen atoms.
        Furthermore, the returned structure contains *AutoDock* atom
        types as ``element`` annotation.
    receptor_coord : ndarray, shape=(m,n,3), dtype=float
        The coordinates for *m* binding modes and *n* atoms
        of the input receptor.
        The models are sorted from best to worst predicted binding
        affinity.
        Missing coordinates due to the removed nonpolar hydrogen
        atoms from flexible side chains are set to *NaN*.
        The output is only meaningful, if flexible side chains were
        defined.
        Otherwise, the coordinates are simply *m* repetitions
        of the input receptor coordinates.
    energies : ndarray, dtype=float
        The predicted binding energies (kcal/mol).
        The energies are sorted from best to worst.
    """

    ligand_models: AtomArrayStack
    ligand_coord: np.ndarray
    flexible_residue_models: AtomArrayStack
    receptor_coord: np.ndarray
    energies: np.ndarray


class VinaApp(LocalApp):
    """
    A handle to *AutoDock Vina*.

    Parameters
    ----------
    path : str, optional
        Path to the ``vina`` binary.

    Examples
    --------

    >>> # A dummy receptor and ligand
    >>> ligand = residue("ASP")
    >>> receptor = atom_array
    >>> app = VinaApp()
    >>> result = app.run(
    ...     ligand, receptor,
    ...     # Binding pocket is in the center of the receptor
    ...     center=centroid(receptor),
    ...     # 20 Å x 20 Å x 20 Å search space
    ...     size=[20, 20, 20],
    ...     # Handle residues 2 and 5 as flexible
    ...     flexible=(receptor.res_id == 2) | (receptor.res_id == 5)
    ... ).result()
    """

    def __init__(self, path: PathLike[str] | str = "vina") -> None:
        super().__init__(path)

    def _format_key(self, key: str) -> str:
        # Vina uses double-dash options and preserves underscores
        return "--" + str(key)

    @command(
        allowed_options=[
            "cpu",
            "exhaustiveness",
            "num_modes",
            "energy_range",
            "min_rmsd",
            "max_evals",
            "spacing",
        ]
    )
    def run(
        self,
        ligand: AtomArray,
        receptor: AtomArray,
        center: np.ndarray,
        size: np.ndarray,
        flexible: np.ndarray | None = None,
        seed: int | None = None,
        **kwargs: Any,
    ) -> CommandSetup[VinaResult]:
        """
        Dock a ligand to a receptor molecule.

        Parameters
        ----------
        ligand : AtomArray
            The structure of the ligand molecule.
            Must have an associated :class:`BondList`.
            An associated ``charge`` annotation is recommended for proper
            calculation of partial charges.
        receptor : AtomArray, shape=(n,)
            The structure of the receptor molecule.
            Must have an associated :class:`BondList`.
            An associated ``charge`` annotation is recommended for proper
            calculation of partial charges.
        center : ndarray, shape=(3,), dtype=float
            The *xyz* coordinates for the center of the search space.
        size : ndarray, shape=(3,), dtype=float
            The size of the search space in *xyz* directions.
        flexible : ndarray, shape=(n,), dtype=bool, optional
            A boolean mask that indicates flexible amino acid side chains
            in `receptor`.
            Each residue, where at least one atom index is ``True`` in
            `flexible`, is considered flexible.
            By default, the receptor has no flexibility.
        seed : int, optional
            The seed for the random number generator, used to make the
            docking run reproducible.
            Must not be ``0``, as *AutoDock Vina* interprets ``0`` as no seed set.
            By default, *Vina* chooses a random seed.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of VinaResult
            A handle to the running docking run.
            Call :meth:`Future.result()` to obtain the
            :class:`VinaResult`.

        Raises
        ------
        ValueError
            If `seed` is ``0``.
        """
        if ligand.bonds is None:
            raise ValueError("The ligand has no associated BondList")
        if receptor.bonds is None:
            raise ValueError("The receptor has no associated BondList")
        if seed == 0:
            raise ValueError(
                "A seed of 0 is interpreted by Vina as a request for a random seed"
            )

        ligand = ligand.copy()
        receptor = receptor.copy()
        is_flexible = flexible is not None

        if flexible is not None:
            flexible_indices = np.where(flexible)[0]
            flex_res_starts = np.unique(
                get_residue_starts_for(receptor, flexible_indices)
            )

        ligand_file = NamedTemporaryFile("w", suffix=".pdbqt", delete=False)
        receptor_file = NamedTemporaryFile("w", suffix=".pdbqt", delete=False)
        receptor_flex_file = NamedTemporaryFile("w", suffix=".pdbqt", delete=False)
        out_file = NamedTemporaryFile("r", suffix=".pdbqt", delete=False)

        # Use different atom ID ranges for atoms in ligand and receptor
        # for unambiguous assignment, if the receptor contains flexible
        # residues
        ligand.set_annotation("atom_id", np.arange(1, ligand.array_length() + 1))
        receptor.set_annotation(
            "atom_id",
            np.arange(
                ligand.array_length() + 1,
                ligand.array_length() + receptor.array_length() + 1,
            ),
        )

        ligand_pdbqt = PDBQTFile()
        # Contains 'true' entries for all atoms that have not been
        # removed from ligand
        ligand_mask = ligand_pdbqt.set_structure(ligand, rotatable_bonds="all")
        ligand_pdbqt.write(ligand_file)
        ligand_file.flush()

        if is_flexible:
            rigid_mask = np.ones(receptor.array_length(), dtype=bool)
            # Contains 'true' entries for all atoms that have not been
            # removed from receptor in flexible side chains
            receptor_mask = np.zeros(receptor.array_length(), dtype=bool)
            for i, start in enumerate(flex_res_starts):
                flex_mask, res_rigid_mask, root = _get_flexible_residue(receptor, start)
                rigid_mask &= res_rigid_mask
                root_in_flex_residue = np.where(
                    np.arange(receptor.array_length())[flex_mask] == root
                )[0][0]
                flex_pdbqt = PDBQTFile()
                receptor_mask[flex_mask] |= flex_pdbqt.set_structure(
                    receptor[flex_mask],
                    rotatable_bonds="all",
                    root=root_in_flex_residue,
                    include_torsdof=False,
                )
                # Enclose each flexible residue
                # with BEGIN_RES and END_RES
                receptor_flex_file.write(f"BEGIN_RES {i}\n")
                flex_pdbqt.write(receptor_flex_file)
                receptor_flex_file.write(f"END_RES {i}\n")
            receptor_flex_file.flush()

            receptor_pdbqt = PDBQTFile()
            receptor_pdbqt.set_structure(
                receptor[rigid_mask],
                rotatable_bonds=None,
                include_torsdof=False,
            )
            receptor_pdbqt.write(receptor_file)
            receptor_file.flush()

        else:
            # These masks are only consulted for flexible receptors, but
            # are kept as arrays here to keep them well-typed
            rigid_mask = np.ones(receptor.array_length(), dtype=bool)
            receptor_mask = np.zeros(receptor.array_length(), dtype=bool)
            receptor_pdbqt = PDBQTFile()
            receptor_pdbqt.set_structure(
                receptor, rotatable_bonds=None, include_torsdof=False
            )
            receptor_pdbqt.write(receptor_file)
            receptor_file.flush()

        parameters: list[CLIParameter] = [
            CLIOption("ligand", ligand_file.name),
            CLIOption("receptor", receptor_file.name),
            CLIOption("out", out_file.name),
            CLIOption("center_x", f"{center[0]:.3f}"),
            CLIOption("center_y", f"{center[1]:.3f}"),
            CLIOption("center_z", f"{center[2]:.3f}"),
            CLIOption("size_x", f"{size[0]:.3f}"),
            CLIOption("size_y", f"{size[1]:.3f}"),
            CLIOption("size_z", f"{size[2]:.3f}"),
        ]
        if is_flexible:
            parameters.append(CLIOption("flex", receptor_flex_file.name))
        if seed is not None:
            parameters.append(CLIOption("seed", seed))

        def evaluate(stdout: bytes, stderr: bytes) -> VinaResult:
            out_pdbqt = PDBQTFile.read(out_file)
            models = out_pdbqt.get_structure()

            n_ligand_atoms = np.count_nonzero(ligand_mask)
            ligand_models = models[..., :n_ligand_atoms]
            flex_models = models[..., n_ligand_atoms:]
            n_models = models.stack_depth()

            remarks = out_pdbqt.get_remarks()
            energies = np.array(
                # VINA RESULT:      -5.8      0.000      0.000
                #                     ^
                [float(remark[12:].split()[0]) for remark in remarks]
            )

            # Ligand coordinates for each binding mode
            ligand_coord = np.full(
                (n_models, ligand.array_length(), 3), np.nan, dtype=np.float32
            )
            ligand_coord[:, ligand_mask] = ligand_models.coord

            # Receptor coordinates for each binding mode
            receptor_coord = np.repeat(
                receptor.coord[np.newaxis, ...], repeats=n_models, axis=0
            )
            if is_flexible:
                # Replace original coordinates with modeled coordinates
                # for the flexible side chains
                # The coordinates from removed atoms are NaN
                receptor_coord[:, ~rigid_mask] = np.nan
                receptor_coord[:, receptor_mask] = flex_models.coord

            return VinaResult(
                ligand_models=ligand_models,
                ligand_coord=ligand_coord,
                flexible_residue_models=flex_models,
                receptor_coord=receptor_coord,
                energies=energies,
            )

        def cleanup() -> None:
            for temp_file in (
                ligand_file,
                receptor_file,
                receptor_flex_file,
                out_file,
            ):
                cleanup_tempfile(temp_file)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
        )


def _get_flexible_residue(
    receptor: AtomArray, residue_start: int
) -> tuple[np.ndarray, np.ndarray, int]:
    residue_indices = np.where(get_residue_masks(receptor, [residue_start])[0])[0]
    root_indices_in_residue = np.isin(receptor.atom_name[residue_indices], ("CA",))
    root_indices = residue_indices[root_indices_in_residue]
    if len(root_indices) == 0:
        raise BadStructureError("Found no CA atom in residue")
    if len(root_indices) > 1:
        raise BadStructureError("Multiple CA atom in residue")
    root_index = root_indices[0]

    # Find the index of the atom connected to root on the flexible
    # side chain (CB)
    if receptor.bonds is None:
        raise ValueError("The receptor has no associated BondList")
    root_connect_indices, _ = receptor.bonds.get_bonds(root_index)
    connected_index = None
    try:
        connected_index = root_connect_indices[
            np.isin(receptor.atom_name[root_connect_indices], ("CB",))
        ][0]
    except IndexError:
        # Residue has no appropriate connection (e.g. in glycine)
        # -> There is no atom in the flexible side chain
        flex_mask = np.zeros(receptor.array_length(), dtype=bool)
        rigid_mask = np.ones(receptor.array_length(), dtype=bool)
        return flex_mask, rigid_mask, root_index

    # Remove the root bond from the bond list
    # to find the atoms involved in the flexible part
    bonds = receptor.bonds.copy()
    bonds.remove_bond(root_index, connected_index)
    flexible_indices = find_connected(bonds, connected_index)
    if root_index in flexible_indices:
        raise BadStructureError(
            "There are multiple connections between the flexible and "
            "rigid part, maybe a cyclic residue like proline was selected"
        )

    flex_mask = np.zeros(receptor.array_length(), dtype=bool)
    flex_mask[flexible_indices] = True
    rigid_mask = ~flex_mask
    # Root index is part of rigid and flexible part
    flex_mask[root_index] = True

    return flex_mask, rigid_mask, root_index
