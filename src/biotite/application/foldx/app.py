# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.foldX"
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
from ...structure.io.pdb import PDBFile
from ...structure.io.pdbx import PDBxFile
from ...structure.residues import get_residue_starts_for, get_residue_masks
from ...structure.bonds import find_connected
from ...structure.error import BadStructureError


class FoldXApp(LocalApp):
    """
    Mutate a protein with *FoldX*

    Parameters
    ----------
    receptor : AtomArray, shape=(n,)
        The structure of the proiten molecule.
        
    mutation : str, in classical mutation format
    subunit  : std, a subunit index in the pdb file
    bin_path : str, optional
        Path to the *FoldX* binary.

    Examples
    --------
    >>> # simple protein
    >>> >>> receptor = atom_array
    >>> app = FoldXApp(receptor,
    ... "A34G+E44G", subunit = "A"
    ... )
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

        self._rotabase_file  = NamedTemporaryFile(
            "w", suffix=".txt", delete=False
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
        """
        Create a temporarily folding file for the mutation

        Parameters
        ----------
        subnut: str
        mutation :str
        """
        entry = [elem[0]+subunit+elem[1:] for elem in mutation.split("+")]
        self._folding_file.write(";".join(entry)+";\n")
        self._folding_file.flush()


    def run(self):
        # Use different atom ID ranges for atoms in ligand and receptor
        # for unambiguous assignment, if the receptor contains flexible
        # residues
        receptor_file = PDBFile()
        receptor_file.set_structure(self._receptor)
        receptor_file.write(self._receptor_file)

        self._receptor_file.flush()

        # set up folding file
        self.setup_folding_file(self._subunit,self._mutation)

        # tempfile
        temp = "/".join(self._receptor_file.name.split("/")[0:-1])

        # set up rotabase - copy to tempfile
        rotabase_path = "/".join(os.path.realpath(__file__).split("/")[0:-1])+"/rotabase.txt"
        #os.popen('cp '+rotabase_path+' '+getcwd()+"/rotabase.txt") 
        os.popen('cp '+rotabase_path+' '+self._rotabase_file.name)

        arguments = [
            "--command", "BuildModel",
            "--pdb", self._receptor_file.name.split("/")[-1],
            "--pdb-dir", temp,
            "--mutant-file", self._folding_file.name,
            "--output-dir", temp,
            "--rotabaseLocation", self._rotabase_file.name,
            #"--output-file", self._mutated_receptor_file.name.split("/")[-1], 
            "--clean-mode", "1",
            "--pdbHydrogens","1"
        ]
        self._output_filename = temp+"/"+self._receptor_file.name.split("/")[-1][:-4]+"_1.pdb"
        self.set_arguments(arguments)

        super().run()
    
    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._receptor_file)
        cleanup_tempfile(self._mutated_receptor_file)
        cleanup_tempfile(self._folding_file)
        cleanup_tempfile(self._rotabase_file)

        # remove rotabase file
        #os.remove(getcwd()+"/rotabase.txt")
        
        # remove output_file 
        os.remove(self._output_filename)

    
    def evaluate(self):
        super().evaluate()
        out_file = PDBFile.read(self._output_filename)
        models = out_file.get_structure(include_bonds = True, model=1)
        print (models)
        self.new_mutant = models
    

    @requires_state(AppState.JOINED)
    def get_mutant(self):
        """
        Get the mutant protein structure

        Returns
        -------
        ligand : AtomArrayStack
            The mutated protein
        
        """
        return self.new_mutant

    @staticmethod
    def mutate(receptor, mutation, bin_path="vina"):
        """
        Mutate the protein with *FoldX BuildModel*.

        This is a convenience function, that wraps the :class:`FoldX`
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
