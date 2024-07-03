# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.mafft"
__author__ = "Patrick Kunzmann"
__all__ = ["MafftApp"]

import os
import re
from biotite.application.application import AppState, requires_state
from biotite.application.msaapp import MSAApp
from biotite.sequence.phylo.tree import Tree

_prefix_pattern = re.compile(r"\d*_")


class MafftApp(MSAApp):
    """
    Perform a multiple sequence alignment using MAFFT.

    Parameters
    ----------
    sequences : list of Sequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the MUSCLE binary.
    matrix : SubstitutionMatrix, optional
        A custom substitution matrix.

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

    def __init__(self, sequences, bin_path="mafft", matrix=None):
        super().__init__(sequences, bin_path, matrix)
        self._tree = None
        self._out_tree_file_name = self.get_input_file_path() + ".tree"

    def run(self):
        args = [
            "--quiet",
            "--auto",
            "--treeout",
            # Get the reordered alignment in order for
            # get_alignment_order() to work properly
            "--reorder",
        ]
        if self.get_seqtype() == "protein":
            args += ["--amino"]
        else:
            args += ["--nuc"]
        if self.get_matrix_file_path() is not None:
            args += ["--aamatrix", self.get_matrix_file_path()]
        args += [self.get_input_file_path()]
        self.set_arguments(args)
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

    def clean_up(self):
        os.remove(self._out_tree_file_name)

    @requires_state(AppState.JOINED)
    def get_guide_tree(self):
        """
        Get the guide tree created for the progressive alignment.

        Returns
        -------
        tree : Tree
            The guide tree.
        """
        return self._tree

    @staticmethod
    def supports_nucleotide():
        return True

    @staticmethod
    def supports_protein():
        return True

    @staticmethod
    def supports_custom_nucleotide_matrix():
        return True

    @staticmethod
    def supports_custom_protein_matrix():
        return True
