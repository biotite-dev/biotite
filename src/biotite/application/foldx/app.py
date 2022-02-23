# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.fold"
__author__ = "Mojmir Mutny"
__all__ = ["FoldXApp"]


import copy
from tempfile import NamedTemporaryFile
import numpy as np
import os 
from os import chdir, getcwd, remove
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...structure.io.pdbqt import PDBQTFile
from ...structure.residues import get_residue_starts_for, get_residue_masks
from ...structure.bonds import find_connected
from ...structure.error import BadStructureError


class FoldXApp(LocalApp):
    """
    Mutate a protein with *FoldX*

    Parameters
    ----------
    receptor : AtomArray, shape=(n,)
        The structure of the receptor molecule.
        Must have an associated :class:`BondList`.
        An associated ``charges`` annotation is recommended for proper
        calculation of partial charges.

    bin_path : str, optional
        Path to the *FoldX* binary.

    Examples
    --------

    """
    def __init__(self, receptor, mutation, subunit = 'B', bin_path="foldx"):

        super().__init__(bin_path)

        self._receptor = receptor.copy()
        self._mutation = mutation
        self._subunit = subunit
        self._seed = None

        self._receptor_file  = NamedTemporaryFile(
            "w", suffix=".pdb", delete=False
        )
        self._mutated_receptor_file  = NamedTemporaryFile(
            "w", suffix=".pdb", delete=False
        )

        self._folding_file  = NamedTemporaryFile(
            "w", suffix=".txt", delete=False, prefix = "individual_list"
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


    def setup_folding_file(self, subunit, mutation):
        entry = [elem[0]+subunit+elem[1:] for elem in mutation.split("+")]
        self._folding_file.write(";".join(entry)+";\n")
        self._folding_file.flush()


    def run(self):
        # Use different atom ID ranges for atoms in ligand and receptor
        # for unambiguous assignment, if the receptor contains flexible
        # residues
        receptor_file = PDBQTFile()
        receptor_file.set_structure(
                self._receptor,
                rotatable_bonds=None,
                include_torsdof=False
            )
        receptor_file.write(self._receptor_file)
        self._receptor_file.flush()

        # set up folding file
        self.setup_folding_file(self._subunit,self._mutation)

        # tempfile
        temp = "/".join(self._receptor_file.name.split("/")[0:-1])

        # set up rotabase - copy to tempfile
        rotabase_path = "/".join(os.path.realpath(__file__).split("/")[0:-1])+"/rotabase.txt"
        os.popen('cp '+rotabase_path+' '+getcwd()+"/rotabase.txt") 
        
        arguments = [
            "--command", "BuildModel",
            "--pdb", self._receptor_file.name.split("/")[-1],
            "--pdb-dir", temp,
            "--mutant-file", self._folding_file.name,
            "--output-dir", temp,
        ]
        self._output_filename = temp+"/"+self._receptor_file.name.split("/")[-1][:-4]+"_1.pdb"
        self.set_arguments(arguments)

        super().run()
    
    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._receptor_file)
        cleanup_tempfile(self._mutated_receptor_file)
        cleanup_tempfile(self._folding_file)

        # remove rotabase file
        os.remove(getcwd()+"/rotabase.txt")
        
        # remove output_file 
        os.remove(self._output_filename)

    
    def evaluate(self):
        super().evaluate()
        out_file = PDBQTFile.read(self._output_filename)
        models = out_file.get_structure()
        self.new_mutant = models
    

    @requires_state(AppState.JOINED)
    def get_mutant(self):
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
        return self.new_mutant

    @staticmethod
    def mutate(receptor, mutation, bin_path="vina"):
        """
        Dock a ligand to a receptor molecule using *AutoDock Vina*.

        This is a convenience function, that wraps the :class:`VinaApp`
        execution.

        Parameters
        ----------

        Returns
        -------
        
        """
        app = FoldXApp( receptor, mutation, bin_path)
        app.start()
        app.join()
        return app.get_mutant()
