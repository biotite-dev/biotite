# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.application.viennarna"
__author__ = "Tom David Müller"
__all__ = ["RNAplotApp"]

import numpy as np
from tempfile import NamedTemporaryFile
from os import remove
from enum import IntEnum
from ..localapp import LocalApp, cleanup_tempfile
from ..application import AppState, requires_state
from ...structure.dotbracket import dot_bracket as dot_bracket_

class RNAplotApp(LocalApp):
    """
    Get coordinates for a 2D representation of any unknotted RNA
    structure using *ViennaRNA's* *RNAplot*.

    The structure has to be provided either in dot bracket notation or
    as a ``ndarray`` of base pairs and the total sequence length.

    Internally this creates a :class:`Popen` instance, which handles
    the execution.

    Parameters
    ----------
    dot_bracket : str, optional (default: None)
        The structure in dot bracket notation.
    base_pairs : ndarray, shape=(n,2), optional (default: None)
        Each row corresponds to the positions of the bases in the
        strand. This parameter is mutually exclusive to ``dot_bracket``.
    length : int, optional (default: None)
        The number of bases in the strand. This parameter is required if
        ``base_pairs`` is given.
    layout_type : RNAplotApp.Layout, optional (default: RNAplotApp.Layout.NAVIEW)
        The layout type according to the *RNAplot* documentation.
    bin_path : str, optional
        Path of the *RNAplot* binary.

    Examples
    --------

    >>> app = RNAplotApp('((..))')
    >>> app.start()
    >>> app.join()
    >>> print(app.get_coordinates())
    [[ -92.50   92.50]
     [ -92.50   77.50]
     [ -90.31   58.24]
     [-109.69   58.24]
     [-107.50   77.50]
     [-107.50   92.50]]
    """

    class Layout(IntEnum):
        """
        This enum type represents the layout type of the plot according
        to the official *RNAplot* orientation.
        """
        RADIAL = 0,
        NAVIEW = 1,
        CIRCULAR = 2,
        RNATURTLE = 3,
        RNAPUZZLER = 4

    def __init__(self, dot_bracket=None, base_pairs=None, length=None,
                 layout_type=Layout.NAVIEW, bin_path="RNAplot"):
        super().__init__(bin_path)

        if dot_bracket is not None:
            self._dot_bracket = dot_bracket
        elif (base_pairs is not None) and (length is not None):
            self._dot_bracket = dot_bracket_(
                base_pairs, length, max_pseudoknot_order = 0
            )[0]
        else:
            raise ValueError(
                "Structure has to be provided in either dot bracket notation "
                "or as base pairs and total sequence length"
            )

        # Get the value of the enum type
        self._layout_type = str(int(layout_type))
        self._in_file  = NamedTemporaryFile("w", suffix=".fold",  delete=False)

    def run(self):
        self._in_file.write("N"*len(self._dot_bracket) + "\n")
        self._in_file.write(self._dot_bracket)
        self._in_file.flush()
        self.set_arguments(
            ["-i", self._in_file.name, "-o", "xrna", "-t", self._layout_type]
        )
        super().run()

    def evaluate(self):
        super().evaluate()
        self._coordinates = np.loadtxt("rna.ss", usecols=(2, 3))

    def clean_up(self):
        super().clean_up()
        cleanup_tempfile(self._in_file)
        remove("rna.ss")

    @requires_state(AppState.CREATED)
    def set_layout_type(self, layout_type):
        """
        Adjust the layout type for the plot according to the *RNAplot*
        documentation.

        Parameters
        ----------
        type : RNAplotApp.Layout
            The layout type.
        """
        self._layout_type = str(layout_type)

    @requires_state(AppState.JOINED)
    def get_coordinates(self):
        """
        Get coordinates for a 2D representation of the input structure.

        Returns
        -------
        coordinates : ndarray, shape=(n,2)
            The 2D coordinates. Each row represents the *x* and *y*
            coordinates for a total sequence length of *n*.

        Examples
        --------

        >>> app = RNAplotApp('((..))')
        >>> app.start()
        >>> app.join()
        >>> print(app.get_coordinates())
        [[ -92.50   92.50]
         [ -92.50   77.50]
         [ -90.31   58.24]
         [-109.69   58.24]
         [-107.50   77.50]
         [-107.50   92.50]]
        """
        return self._coordinates

    @staticmethod
    def compute_coordinates(
        dot_bracket=None, base_pairs=None, length=None,
        layout_type=Layout.NAVIEW, bin_path="RNAplot"
    ):
        """
        Get coordinates for a 2D representation of any unknotted RNA
        structure using *ViennaRNA's* *RNAplot*.

        The structure has to be provided either in dot bracket notation
        or as a ``ndarray`` of base pairs and the total sequence length.

        This is a convenience function, that wraps the
        :class:`RNAplotApp` execution.

        Parameters
        ----------
        dot_bracket : str, optional (default: None)
            The structure in dot bracket notation.
        base_pairs : ndarray, shape=(n,2), optional (default: None)
            Each row corresponds to the positions of the bases in the
            strand. This parameter is mutually exclusive to
            ``dot_bracket``.
        length : int, optional (default: None)
            The number of bases in the strand. This parameter is
            required if ``base_pairs`` is given.
        bin_path : str, optional
            Path of the *RNAplot* binary.

        Returns
        -------
        coordinates : ndarray, shape=(n,2)
            The 2D coordinates. Each row represents the *x* and *y*
            coordinates for a total sequence length of *n*.
        """
        app = RNAplotApp(dot_bracket=dot_bracket, base_pairs=base_pairs,
                         length=length, layout_type=layout_type,
                         bin_path=bin_path)
        app.start()
        app.join()
        return app.get_coordinates()