# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.autodock"
__author__ = "Patrick Kunzmann"
__all__ = ["VinaApp"]

import copy
from tempfile import NamedTemporaryFile
import numpy as np
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...structure.io.pdbqt import PDBQTFile
from ...structure.residues import get_residue_starts_for, get_residue_masks
from ...structure.bonds import find_connected
from ...structure.error import BadStructureError


class VinaApp(LocalApp):
    """
    Dock a ligand to a receptor molecule using *AutoDock Vina*.

    Parameters
    ----------
    ligand : AtomArray
        The structure of the receptor molecule.
        Must have an associated :class:`BondList`.
        An associated ``charges`` annotation is recommended for proper
        calculation of partial charges.
    receptor : AtomArray, shape=(n,)
        The structure of the receptor molecule.
        Must have an associated :class:`BondList`.
        An associated ``charges`` annotation is recommended for proper
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
    bin_path : str, optional
        Path to the *Vina* binary.

    Examples
    --------

    >>> # A dummy receptor and ligand
    >>> ligand = residue("ASP")
    >>> receptor = atom_array
    >>> app = VinaApp(
    ...     ligand, receptor,
    ...     # Binding pocket is in the center of the receptor
    ...     center=centroid(receptor),
    ...     # 20 Å x 20 Å x 20 Å search space
    ...     size=[20, 20, 20],
    ...     # Handle residues 2 and 5 as flexible
    ...     flexible=(receptor.res_id == 2) | (receptor.res_id == 5)
    ... )
    """
    def __init__(self, ligand, receptor, center, size, flexible=None,
                 bin_path="vina"):
        super().__init__(bin_path)

        if ligand.bonds is None:
            raise ValueError("The ligand has no associated BondList")
        if receptor.bonds is None:
            raise ValueError("The receptor has no associated BondList")

        self._ligand = ligand.copy()
        self._receptor = receptor.copy()
        self._center = copy.deepcopy(center)
        self._size = copy.deepcopy(size)
        self._is_flexible = flexible is not None
        self._seed = None
        self._exhaustiveness = None
        self._number = None
        self._energy_range = None

        if self._is_flexible:
            flexible_indices = np.where(flexible)[0]
            self._flex_res_starts = np.unique(get_residue_starts_for(
                receptor, flexible_indices
            ))
        
        self._ligand_file  = NamedTemporaryFile(
            "w", suffix=".pdbqt", delete=False
        )
        self._receptor_file  = NamedTemporaryFile(
            "w", suffix=".pdbqt", delete=False
        )
        self._receptor_flex_file  = NamedTemporaryFile(
            "w", suffix=".pdbqt", delete=False
        )
        self._out_file  = NamedTemporaryFile(
            "r", suffix=".pdbqt", delete=False
        )
    
    @requires_state(AppState.CREATED)
    def set_seed(self, seed):
        """
        Fix the seed for the random number generator to get
        reproducible results.

        By default, the seed is chosen randomly.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.
        """
        self._seed = seed
    
    @requires_state(AppState.CREATED)
    def set_exhaustiveness(self, exhaustiveness):
        """
        Set the *exhaustiveness* parameter for *Vina*.

        A higher exhaustiveness may lead to better docking results, but
        also increases the computation time.
        By default, the exhaustiveness is ``8``.

        Parameters
        ----------
        exhaustiveness : int
            The value for the exhaustiveness parameter.
            Must be greater than 0.
        """
        self._exhaustiveness = exhaustiveness
    
    @requires_state(AppState.CREATED)
    def set_max_number_of_models(self, number):
        """
        Set the maximum number of binding modes to generate.

        *Vina* may generate less modes, if the docking process does
        not find enough distinct conformations.
        By default, the maximum number is ``9``.

        Parameters
        ----------
        number : int
            The maximum number of generated modes/models.
        """
        self._number = number
    
    @requires_state(AppState.CREATED)
    def set_energy_range(self, energy_range):
        """
        Set the maximum energy range of the generated models.

        *Vina* will ignore binding modes if the difference between this
        mode and the best mode is greater than this value.
        By default, the range is ``3.0``.

        Parameters
        ----------
        number : float
            The energy range (kcal/mol).
        """
        self._energy_range = energy_range

    def run(self):
        # Use different atom ID ranges for atoms in ligand and receptor
        # for unambiguous assignment, if the receptor contains flexible
        # residues
        self._ligand.set_annotation("atom_id", np.arange(
            1,
            self._ligand.array_length() + 1
        ))
        self._receptor.set_annotation("atom_id", np.arange(
            self._ligand.array_length() + 1,
            self._ligand.array_length() + self._receptor.array_length() + 1
        ))

        ligand_file = PDBQTFile()
        # Contains 'true' entries for all atoms that have not been 
        # removed from ligand
        self._ligand_mask = ligand_file.set_structure(
            self._ligand,
            rotatable_bonds="all"
        )
        ligand_file.write(self._ligand_file)
        self._ligand_file.flush()
        
        if self._is_flexible:
            self._rigid_mask = np.ones(
                self._receptor.array_length(), dtype=bool
            )
            # Contains 'true' entries for all atoms that have not been 
            # removed from receptor in flexible side chains
            self._receptor_mask = np.zeros(
                self._receptor.array_length(), dtype=bool
            )
            for i, start in enumerate(self._flex_res_starts):
                flex_mask, rigid_mask, root = self._get_flexible_residue(start)
                self._rigid_mask &= rigid_mask
                root_in_flex_residue = np.where(
                    np.arange(self._receptor.array_length())[flex_mask] == root
                )[0][0]
                flex_file = PDBQTFile()
                self._receptor_mask[flex_mask] |= flex_file.set_structure(
                    self._receptor[flex_mask],
                    rotatable_bonds="all",
                    root=root_in_flex_residue,
                    include_torsdof=False
                )
                # Enclose each flexible residue
                # with BEGIN_RES and END_RES
                self._receptor_flex_file.write(f"BEGIN_RES {i}\n")
                flex_file.write(self._receptor_flex_file)
                self._receptor_flex_file.write(f"END_RES {i}\n")
            self._receptor_flex_file.flush()

            receptor_file = PDBQTFile()
            receptor_file.set_structure(
                self._receptor[self._rigid_mask],
                rotatable_bonds=None,
                include_torsdof=False
            )
            receptor_file.write(self._receptor_file)
            self._receptor_file.flush()

        else:
            receptor_file = PDBQTFile()
            receptor_file.set_structure(
                self._receptor,
                rotatable_bonds=None,
                include_torsdof=False
            )
            receptor_file.write(self._receptor_file)
            self._receptor_file.flush()

        arguments = [
            "--ligand", self._ligand_file.name,
            "--receptor", self._receptor_file.name,
            "--out", self._out_file.name,
            "--center_x", f"{self._center[0]:.3f}",
            "--center_y", f"{self._center[1]:.3f}",
            "--center_z", f"{self._center[2]:.3f}",
            "--size_x", f"{self._size[0]:.3f}",
            "--size_y", f"{self._size[1]:.3f}",
            "--size_z", f"{self._size[2]:.3f}",
        ]
        if self._seed is not None:
            arguments.extend(["--seed", str(self._seed)])
        if self._exhaustiveness is not None:
            arguments.extend(["--exhaustiveness", str(self._exhaustiveness)])
        if self._number is not None:
            arguments.extend(["--num_modes", str(self._number)])
        if self._energy_range is not None:
            arguments.extend(["--energy_range", str(self._energy_range)])
        if self._is_flexible:
            arguments.extend(["--flex", str(self._receptor_flex_file.name)])

        self.set_arguments(arguments)
        super().run()
    
    def evaluate(self):
        super().evaluate()
        out_file = PDBQTFile.read(self._out_file)
        
        models = out_file.get_structure()

        n_ligand_atoms = np.count_nonzero(self._ligand_mask)
        self._ligand_models = models[..., :n_ligand_atoms]
        self._flex_models = models[..., n_ligand_atoms:]
        self._n_models = models.stack_depth()
        
        remarks = out_file.get_remarks()
        self._energies = np.array(
            # VINA RESULT:      -5.8      0.000      0.000
            #                     ^
            [float(remark[12:].split()[0]) for remark in remarks]
        )
    
    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._ligand_file)
        cleanup_tempfile(self._receptor_file)
        cleanup_tempfile(self._receptor_flex_file)
        cleanup_tempfile(self._out_file)
    
    @requires_state(AppState.JOINED)
    def get_energies(self):
        """
        Get the predicted binding energy for each generated binding
        mode.

        Returns
        -------
        energies : ndarray, dtype=float
            The predicted binding energies (kcal/mol).
            The energies are sorted from best to worst.
        """
        return self._energies

    @requires_state(AppState.JOINED)
    def get_ligand_models(self):
        """
        Get the ligand structure with the conformations for each 
        generated binding mode.

        Returns
        -------
        ligand : AtomArrayStack
            The docked ligand.
            Each model corresponds to one binding mode.
            The models are sorted from best to worst predicted binding
            affinity.
        
        Notes
        -----
        The returned structure may contain less atoms than the input
        structure, as *Vina* removes nonpolar hydrogen atoms.
        Furthermore, the returned structure contains *AutoDock* atom
        types as ``element`` annotation.
        """
        return self._ligand_models

    @requires_state(AppState.JOINED)
    def get_ligand_coord(self):
        """
        Get the ligand coordinates for each generated binding mode.

        Returns
        -------
        coord : ndarray, shape=(m,n,3), dtype=float
            The coordinates for *m* binding modes and *n* atoms
            of the input ligand.
            The models are sorted from best to worst predicted binding
            affinity.
            Missing coordinates due to the removed nonpolar hydrogen
            atoms are set to *NaN*.
        """
        coord = np.full(
            (self._n_models, self._ligand.array_length(), 3),
            np.nan, dtype=np.float32
        )
        coord[:, self._ligand_mask] = self._ligand_models.coord
        return coord
    
    @requires_state(AppState.JOINED)
    def get_flexible_residue_models(self):
        """
        Get the structure for the flexible side chains with the
        conformations for each generated binding mode.

        If no flexible side chains were defined, the returned
        :class:`AtomArrayStack` contains no atoms.

        Returns
        -------
        side_chains : AtomArrayStack
            The docked side chains.
            Each model corresponds to one binding mode.
            The models are sorted from best to worst predicted binding
            affinity.
        
        Notes
        -----
        The returned structure may contain less atoms than the input
        structure, as *Vina* removes nonpolar hydrogen atoms.
        Furthermore, the returned structure contains *AutoDock* atom
        types as ``element`` annotation.
        """
        return self._flex_models

    @requires_state(AppState.JOINED)
    def get_receptor_coord(self):
        """
        Get the get_receptor_coord coordinates for each generated
        binding mode.

        Returns
        -------
        coord : ndarray, shape=(m,n,3), dtype=float
            The coordinates for *m* binding modes and *n* atoms
            of the input receptor.
            The models are sorted from best to worst predicted binding
            affinity.
            Missing coordinates due to the removed nonpolar hydrogen
            atoms from flexible side chains are set to *NaN*.
        
        Notes
        -----
        The output is only meaningful, if flexible side chains were
        defined.
        Otherwise, the returned coordinates are simply *m* repetitions
        of the input receptor coordinates.
        """
        coord = np.repeat(
            self._receptor.coord[np.newaxis, ...],
            repeats=self._n_models, axis=0
        )
        if self._is_flexible:
            # Replace original coordinates with modeled coordinates
            # for the the flexible side chains
            # The coordinates from removed atoms are NaN
            coord[:, ~self._rigid_mask] = np.nan
            coord[:, self._receptor_mask] = self._flex_models.coord
        return coord

    def _get_flexible_residue(self, residue_start):
        residue_indices = np.where(
            get_residue_masks(self._receptor, [residue_start])[0]
        )[0]
        root_indices_in_residue = np.isin(
            self._receptor.atom_name[residue_indices], ("CA",)
        )
        root_indices = residue_indices[root_indices_in_residue]
        if len(root_indices) == 0:
            raise BadStructureError("Found no CA atom in residue")
        if len(root_indices) > 1:
            raise BadStructureError("Multiple CA atom in residue")
        root_index = root_indices[0]

        # Find the index of the atom connected to root on the flexible
        # side chain (CB)
        root_connect_indices, _ = self._receptor.bonds.get_bonds(root_index)
        connected_index = None
        try:
            connected_index = root_connect_indices[np.isin(
                self._receptor.atom_name[root_connect_indices], ("CB",)
            )][0]
        except IndexError:
            # Residue has no appropriate connection (e.g. in glycine)
            # -> There is no atom in the flexible side chain
            flex_mask = np.zeros(self._receptor.array_length(), dtype=bool)
            rigid_mask = np.ones(self._receptor.array_length(), dtype=bool)
            return flex_mask, rigid_mask, root_index
        
        # Remove the root bond from the bond list
        # to find the atoms involved in the flexible part
        bonds = self._receptor.bonds.copy()
        bonds.remove_bond(root_index, connected_index)
        flexible_indices = find_connected(bonds, connected_index)
        if root_index in flexible_indices:
            raise BadStructureError(
                "There are multiple connections between the flexible and "
                "rigid part, maybe a cyclic residue like proline was selected" 
            )

        flex_mask = np.zeros(self._receptor.array_length(), dtype=bool)
        flex_mask[flexible_indices] = True
        rigid_mask = ~flex_mask
        # Root index is part of rigid and flexible part
        flex_mask[root_index] = True

        return flex_mask, rigid_mask, root_index
    

    @staticmethod
    def dock(ligand, receptor, center, size, flexible=None, bin_path="vina"):
        """
        Dock a ligand to a receptor molecule using *AutoDock Vina*.

        This is a convenience function, that wraps the :class:`VinaApp`
        execution.

        Parameters
        ----------
        ligand : AtomArray
            The structure of the receptor molecule.
            Must have an associated :class:`BondList`.
            An associated ``charges`` annotation is recommended for proper
            calculation of partial charges.
        receptor : AtomArray, shape=(n,)
            The structure of the receptor molecule.
            Must have an associated :class:`BondList`.
            An associated ``charges`` annotation is recommended for proper
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
        bin_path : str, optional
            Path to the *Vina* binary.

        Returns
        -------
        coord : ndarray, shape=(m,n,3), dtype=float
            The docked ligand coordinates for *m* binding modes and
            *n* atoms of the input ligand.
            The models are sorted from best to worst predicted binding
            affinity.
            Missing coordinates due to the removed nonpolar hydrogen
            atoms are set to *NaN*.
        energies : ndarray, shape=(m,), dtype=float
            The corresponding predicted binding energies (kcal/mol).
        """
        app = VinaApp(ligand, receptor, center, size, flexible, bin_path)
        app.start()
        app.join()
        return app.get_ligand_coord(), app.get_energies()
