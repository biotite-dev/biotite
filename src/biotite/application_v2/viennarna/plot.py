# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.viennarna"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = ["PlotApp"]

import os
import tempfile
from enum import IntEnum
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
from biotite.structure.dotbracket import dot_bracket as dot_bracket_
from biotite.typing import C2, N, NDArray2


class PlotApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAplot``.

    The structure has to be provided either in dot bracket notation or
    as a ``ndarray`` of base pairs and the total sequence length.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAplot`` binary.

    Examples
    --------

    >>> coordinates = PlotApp().run("((..))").result()
    >>> print(coordinates)
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
        to the official ``RNAplot`` orientation.
        """

        RADIAL = 0
        NAVIEW = 1
        CIRCULAR = 2
        RNATURTLE = 3
        RNAPUZZLER = 4

    def __init__(self, path: PathLike[str] | str = "RNAplot") -> None:
        super().__init__(path)

    @command
    def run(
        self,
        dot_bracket: str | None = None,
        base_pairs: NDArray2[N, C2, np.integer] | None = None,
        length: int | None = None,
        layout_type: Layout = Layout.NAVIEW,
        **kwargs: Any,
    ) -> CommandSetup[NDArray2[N, C2, np.floating]]:
        """
        Get coordinates for a 2D representation of an RNA structure.

        Parameters
        ----------
        dot_bracket : str, optional
            The structure in dot bracket notation.
        base_pairs : ndarray, shape=(n,2), optional
            Each row corresponds to the positions of the bases in the
            strand.
            This parameter is mutually exclusive to `dot_bracket`.
        length : int, optional
            The number of bases in the strand.
            This parameter is required if `base_pairs` is given.
        layout_type : PlotApp.Layout, optional
            The layout type according to the ``RNAplot`` documentation.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ndarray, shape=(n,2), dtype=float
            A handle to the running computation.
            Call :meth:`Future.result()` to obtain the 2D coordinates,
            where each row holds the *x* and *y* coordinate of one base.
        """
        if dot_bracket is not None:
            structure = dot_bracket
        elif (base_pairs is not None) and (length is not None):
            structure = dot_bracket_(base_pairs, length, max_pseudoknot_order=0)[0]
        else:
            raise ValueError(
                "Structure has to be provided in either dot bracket notation "
                "or as base pairs and total sequence length"
            )

        in_file = NamedTemporaryFile("w", suffix=".fold", delete=False)
        in_file.write("N" * len(structure) + "\n")
        in_file.write(structure)
        in_file.flush()

        # RNAplot writes 'rna.ss' into the working directory
        # -> Use a dedicated temporary directory to avoid polluting the CWD
        exec_dir = tempfile.TemporaryDirectory()

        parameters: list[CLIParameter] = [
            CLIOption("i", in_file.name),
            CLIOption("output_format", "xrna"),
            CLIOption("t", int(layout_type)),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> NDArray2[N, C2, np.floating]:
            return np.loadtxt(os.path.join(exec_dir.name, "rna.ss"), usecols=(2, 3))

        def cleanup() -> None:
            cleanup_tempfile(in_file)
            exec_dir.cleanup()

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=cleanup,
            exec_dir=exec_dir.name,
        )
