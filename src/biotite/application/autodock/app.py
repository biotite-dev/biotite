# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.autodock"
__author__ = "Patrick Kunzmann"
__all__ = ["VinaApp"]

import copy
from tempfile import NamedTemporaryFile
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...structure.io.pdbqt import PDBQTFile
import numpy as np


class VinaApp(LocalApp):
    def __init__(self, ligand, receptor, center, size, bin_path="vina"):
        super().__init__(bin_path)

        if ligand.bonds is None:
            raise ValueError("The ligand has no associated BondList")
        if receptor.bonds is None:
            raise ValueError("The receptor has no associated BondList")

        self._ligand = ligand.copy()
        self._receptor = receptor.copy()
        self._center = copy.deepcopy(center)
        self._size = copy.deepcopy(size)
        self._seed = None
        self._exhaustiveness = None
        self._number = None
        self._energy_range = None
        
        self._ligand_file  = NamedTemporaryFile(
            "w", suffix=".pdbqt", delete=False
        )
        self._receptor_file  = NamedTemporaryFile(
            "w", suffix=".pdbqt", delete=False
        )
        self._out_file  = NamedTemporaryFile(
            "r", suffix=".pdbqt", delete=False
        )
    
    def set_seed(self, seed):
        self._seed = seed
    
    def set_exhaustiveness(self, exhaustiveness):
        self._exhaustiveness = exhaustiveness
    
    def set_number_of_models(self, number):
        self._number = number
    
    def set_energy_range(self, energy_range):
        self._energy_range = energy_range

    def run(self):
        ligand_file = PDBQTFile()
        self._mask = ligand_file.set_structure(
            self._ligand,
            rotatable_bonds="all"
        )
        ligand_file.write(self._ligand_file)
        self._ligand_file.flush()

        receptor_file = PDBQTFile()
        receptor_file.set_structure(
            self._receptor,
            rotatable_bonds=None
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

        self.set_arguments(arguments)
        super().run()
    
    def evaluate(self):
        super().evaluate()
        out_file = PDBQTFile.read(self._out_file)
        
        self._models = out_file.get_structure()
        self._n_models = self._models.stack_depth()
        
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
        cleanup_tempfile(self._out_file)
    
    @requires_state(AppState.JOINED)
    def get_energies(self):
        return self._energies

    @requires_state(AppState.JOINED)
    def get_models(self):
        return self.self._models

    @requires_state(AppState.JOINED)
    def get_coord(self):
        coord = np.full(
            (self._n_models, self._ligand.array_length(), 3),
            np.nan, dtype=np.float32
        )
        coord[:, self._mask] = self._models.coord
        return coord

    
    @staticmethod
    def dock(ligand, receptor, center, size, bin_path="vina"):
        app = VinaApp(ligand, receptor, center, size, bin_path)
        app.start()
        app.join()
        return app.get_coord(), app.get_energies()
