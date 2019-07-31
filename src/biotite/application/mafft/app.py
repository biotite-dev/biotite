# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["MafftApp"]

import re
from ..msaapp import MSAApp
from ..application import AppState, requires_state
from ...sequence.sequence import Sequence
from ...sequence.seqtypes import NucleotideSequence, ProteinSequence
from ...sequence.io.fasta.file import FastaFile
from ...sequence.align.alignment import Alignment
from ...sequence.phylo.tree import Tree


_prefix_pattern = re.compile("._")



class MafftApp(MSAApp):
    """
    Perform a multiple sequence alignment using MAFFT.
    
    Parameters
    ----------
    sequences : iterable object of ProteinSequence or NucleotideSequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MUSCLE binary.
    
    Examples
    --------

    >>> seq1 = ProteinSequence("BIQTITE")
    >>> seq2 = ProteinSequence("TITANITE")
    >>> seq3 = ProteinSequence("BISMITE")
    >>> seq4 = ProteinSequence("IQLITE")
    >>> app = MafftApp([seq1, seq2, seq3, seq4])
    >>> app.start()
    >>> app.join()
    >>> alignment = app.get_alignment()
    >>> print(alignment)
    -BIQTITE
    TITANITE
    -BISMITE
    --IQLITE
    """
    
    def __init__(self, sequences, bin_path="mafft"):
        super().__init__(sequences, bin_path)
        self._tree = None
        self._out_tree_file_name = self.get_input_file_path() + ".tree"
    
    def run(self):
        self.set_arguments(
            ["--auto",
             # Get the reordered alignment in order for
             # get_alignment_order() to work properly 
             "--reorder",
             "--treeout",
             self.get_input_file_path()]
        )
        super().run()
    
    def evaluate(self):
        with open(self.get_output_file_path(), "w") as f:
            # MAFFT outputs alignment to stdout
            # -> write stdout to output file name
            f.write(self.get_stdout())
        super().evaluate()
        with open(self._out_tree_file_name, "r") as file:
            raw_newick = file.read().replace("\n", "")
            # Mafft uses sequences label in the form '<n>_<seqname>'
            # Only the <seqname> is required
            # -> remove the '<n>_' prefix
            newick = re.sub(_prefix_pattern, "", raw_newick)
            self._tree = Tree.from_newick(newick)
    
    @requires_state(AppState.JOINED)
    def get_guide_tree(self):
        return self._tree
