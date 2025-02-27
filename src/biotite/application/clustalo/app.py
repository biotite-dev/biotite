# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.clustalo"
__author__ = "Patrick Kunzmann"
__all__ = ["ClustalOmegaApp"]

from tempfile import NamedTemporaryFile
import numpy as np
from biotite.application.application import AppState, requires_state
from biotite.application.localapp import cleanup_tempfile
from biotite.application.msaapp import MSAApp
from biotite.sequence.phylo.tree import Tree


class ClustalOmegaApp(MSAApp):
    """
    Perform a multiple sequence alignment using Clustal-Omega.

    Parameters
    ----------
    sequences : list of ProteinSequence or NucleotideSequence
        The sequences to be aligned.
    bin_path : str, optional
        Path of the Custal-Omega binary.
    matrix : None
        This parameter is used for compatibility reasons and is ignored.

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

    def __init__(self, sequences, bin_path="clustalo", matrix=None):
        super().__init__(sequences, bin_path, None)
        self._seq_count = len(sequences)
        self._mbed = True
        self._dist_matrix = None
        self._tree = None
        self._in_dist_matrix_file = NamedTemporaryFile("w", suffix=".mat", delete=False)
        self._out_dist_matrix_file = NamedTemporaryFile(
            "r", suffix=".mat", delete=False
        )
        self._in_tree_file = NamedTemporaryFile("w", suffix=".tree", delete=False)
        self._out_tree_file = NamedTemporaryFile("r", suffix=".tree", delete=False)

    def run(self):
        args = [
            "--in",
            self.get_input_file_path(),
            "--out",
            self.get_output_file_path(),
            # The temporary files are already created
            # -> tell Clustal to overwrite these empty files
            "--force",
            # Tree order for get_alignment_order() to work properly
            "--output-order=tree-order",
        ]
        if self.get_seqtype() == "protein":
            args += ["--seqtype", "Protein"]
        else:
            args += ["--seqtype", "DNA"]
        if self._tree is None:
            # ClustalOmega does not like when a tree is set
            # as input and output#
            # -> Only request tree output when not tree is input
            args += [
                "--guidetree-out",
                self._out_tree_file.name,
            ]
        if not self._mbed:
            args += ["--full", "--distmat-out", self._out_dist_matrix_file.name]
        if self._dist_matrix is not None:
            # Add the sequence names (0, 1, 2, 3 ...) as first column
            dist_matrix_with_index = np.concatenate(
                (np.arange(self._seq_count)[:, np.newaxis], self._dist_matrix), axis=1
            )
            np.savetxt(
                self._in_dist_matrix_file.name,
                dist_matrix_with_index,
                # The first line contains the amount of sequences
                comments="",
                header=str(self._seq_count),
                # The sequence indices are integers, the rest are floats
                fmt=["%d"] + ["%.5f"] * self._seq_count,
            )
            args += ["--distmat-in", self._in_dist_matrix_file.name]
        if self._tree is not None:
            self._in_tree_file.write(str(self._tree))
            self._in_tree_file.flush()
            args += ["--guidetree-in", self._in_tree_file.name]
        self.set_arguments(args)
        super().run()

    def evaluate(self):
        super().evaluate()
        if not self._mbed:
            self._dist_matrix = np.loadtxt(
                self._out_dist_matrix_file.name,
                # The first row only contains the number of sequences
                skiprows=1,
                dtype=float,
            )
            # The first column contains only the name of the
            # sequences, in this case 0, 1, 2, 3 ...
            # -> Omit the first column
            self._dist_matrix = self._dist_matrix[:, 1:]
        # Only read output tree if no tree was input
        if self._tree is None:
            self._tree = Tree.from_newick(self._out_tree_file.read().replace("\n", ""))

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_dist_matrix_file)
        cleanup_tempfile(self._out_dist_matrix_file)
        cleanup_tempfile(self._in_tree_file)
        cleanup_tempfile(self._out_tree_file)

    @requires_state(AppState.CREATED)
    def full_matrix_calculation(self):
        """
        Use full distance matrix for guide-tree calculation, equivalent
        to the ``--full`` option.

        This makes the distance matrix calculation slower than using the
        default *mBed* heuristic.
        """
        self._mbed = False

    @requires_state(AppState.CREATED)
    def set_distance_matrix(self, matrix):
        """
        Set the pairwise sequence distances, the program should use to
        calculate the guide tree.

        Parameters
        ----------
        matrix : ndarray, shape=(n,n), dtype=float
            The pairwise distances.
        """
        if matrix.shape != (self._seq_count, self._seq_count):
            raise ValueError(
                f"Matrix with shape {matrix.shape} is not sufficient for "
                f"{self._seq_count} sequences"
            )
        self._dist_matrix = matrix.astype(float, copy=False)

    @requires_state(AppState.JOINED)
    def get_distance_matrix(self):
        """
        Get the pairwise sequence distances the program used to
        calculate the guide tree.

        Returns
        -------
        matrix : ndarray, shape=(n,n), dtype=float
            The pairwise distances.
        """
        if self._mbed:
            raise ValueError(
                "Getting the distance matrix requires 'full_matrix_calculation()'"
            )
        return self._dist_matrix

    @requires_state(AppState.CREATED)
    def set_guide_tree(self, tree):
        """
        Set the guide tree, the program should use for the
        progressive alignment.

        Parameters
        ----------
        tree : Tree
            The guide tree.
        """
        if self._seq_count != len(tree):
            raise ValueError(
                f"Tree with {len(tree)} leaves is not sufficient for "
                "{self._seq_count} sequences, must be equal"
            )
        self._tree = tree

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
        return False

    @staticmethod
    def supports_custom_protein_matrix():
        return False
