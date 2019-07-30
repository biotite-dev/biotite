# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["ClustalOmegaApp"]

from ...temp import temp_file
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.alignment import Alignment
from ...sequence.phylo import Tree
from ..msaapp import MSAApp
from ..application import AppState, requires_state


class ClustalOmegaApp(MSAApp):
    """
    Perform a multiple sequence alignment using Clustal-Omega.
    
    Parameters
    ----------
    sequences : iterable object of ProteinSequence or NucleotideSequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the Custal-Omega binary.
    
    Examples
    --------

    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("TITANITE")
    >>> seq3 = ProteinSequence("BISMITE")
    >>> seq4 = ProteinSequence("IQLITE")
    >>> app = ClustalOmegaApp([seq1, seq2, seq3, seq4])
    >>> app.start()
    >>> app.join()
    >>> alignment = app.get_alignment()
    >>> print(alignment)
    -BIQTITE
    TITANITE
    -BISMITE
    --IQLITE
    """
    
    def __init__(self, sequences, bin_path="clustalo"):
        if isinstance(sequences[0], NucleotideSequence):
            self._seqtype = "DNA"
        else:
            self._seqtype = "Protein"
        self._seq_count = len(sequences)
        self._mbed = True
        self._dist_matrix = None
        self._tree = None
        self._in_dist_matrix_file_name = temp_file("mat")
        self._out_dist_matrix_file_name = temp_file("mat")
        self._in_tree_file_name = temp_file("tree")
        self._out_tree_file_name = temp_file("tree")
        super().__init__(sequences, bin_path)
    
    def run(self):
        args = [
            "--in", self.get_input_file_path(),
            "--out", self.get_output_file_path(),
            "--seqtype", self._seqtype,
            # Tree order for get_alignment_order() to work properly 
            "--output-order=tree-order",
            "--guidetree-out", self._out_tree_file_name,
        ]
        if not self._mbed:
            args += [
                "--full",
                "--distmat-out", self._out_dist_matrix_file_name
            ]
        if self._dist_matrix is not None:
            pass
        if self._tree is not None:
            with open(self._in_tree_file_name, "w") as file:
                file.write(str(self._tree))
            args += ["--guidetree-in", self._in_tree_file_name]
        self.set_arguments(args)
        super().run()
    
    def evaluate(self):
        super().evaluate()
        if not self._mbed:
            with open(self._out_dist_matrix_file_name, "r") as file:
                self._dist_matrix = file.read()
        with open(self._out_tree_file_name, "r") as file:
            self._tree = Tree.from_newick(file.read().replace("\n", ""))
    
    @requires_state(AppState.CREATED)
    def full_matrix_calculation(self):
        self._mbed = False
    
    @requires_state(AppState.CREATED)
    def set_distance_matrix(self, matrix):
        self._dist_matrix = matrix
    
    @requires_state(AppState.JOINED)
    def get_distance_matrix(self):
        if self._mbed:
            raise ValueError(
                "Getting the distance matrix requires "
                "'full_matrix_calculation()'"
            )
        return self._dist_matrix
    
    @requires_state(AppState.CREATED)
    def set_guide_tree(self, tree):
        if self._seq_count != len(tree):
            raise ValueError(
                f"Tree with {len(tree)} leaves is not sufficient for "
                "{self._seq_count} sequences, must be equal"
            )
        self._tree = tree
    
    @requires_state(AppState.JOINED)
    def get_guide_tree(self):
        return self._tree
