# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.viennarna"
__author__ = "Tom David Müller, Patrick Kunzmann"
__all__ = [
    "FoldApp",
    "SuboptApp",
    "PKplexApp",
    "LfoldApp",
    "CofoldApp",
    "MultifoldApp",
    "DuplexApp",
    "PlexApp",
    "AlifoldApp",
    "LalifoldApp",
    "AliduplexApp",
    "EvalApp",
]

import re
import tempfile
import warnings
from collections.abc import Sequence
from os import PathLike
from tempfile import NamedTemporaryFile
from typing import Any
import numpy as np
from biotite.application_v2.localapp import (
    CLIArgument,
    CLIFlag,
    CLIOption,
    CLIParameter,
    CommandSetup,
    LocalApp,
    cleanup_tempfile,
    command,
)
from biotite.application_v2.viennarna.result import (
    CofoldedAlignment,
    CofoldedRNA,
    FoldedAlignment,
    FoldedRNA,
)
from biotite.application_v2.viennarna.util import build_constraint_string
from biotite.sequence.align.alignment import Alignment
from biotite.sequence.io.clustal import ClustalFile
from biotite.sequence.io.clustal import set_alignment as set_clustal_alignment
from biotite.sequence.io.fasta import FastaFile, set_alignment
from biotite.sequence.seqtypes import NucleotideSequence

# Energy-model command line options that are safe to expose as extra options
_ENERGY_OPTIONS = [
    "noLP",
    "noGU",
    "noClosingGU",
    "dangles",
    "paramFile",
    "salt",
    "nsp",
    "energyModel",
]


# A structure token consists exclusively of dot-bracket characters and the
# strand separator '&'; this excludes echoed sequences (which contain
# nucleotide letters) and pure energy lines (which contain digits/'-')
_STRUCTURE_TOKEN_CHARS = set(".()[]{}<>&")
# RNAeval only scores nested '()' base pairs; every other bracket represents a
# pseudoknot that is not part of its energy model
_NESTED_STRUCTURE_CHARS = set(".()&")
_FLOAT_PATTERN = re.compile(r"[-+]?\d+\.\d+")
_RANGE_PATTERN = re.compile(r"(\d+),\s*(\d+)\s*:\s*(\d+),\s*(\d+)")


class FoldApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAfold`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAfold`` binary.

    Examples
    --------

    >>> sequence = NucleotideSequence("CGACGTAGATGCTAGCTGACTCGATGC")
    >>> result = FoldApp().run(sequence).result()
    >>> print(result.free_energy)
    -1.3
    >>> print(result.dot_bracket)
    (((.((((.......)).)))))....
    >>> print(result.base_pairs())
    [[ 0 22]
     [ 1 21]
     [ 2 20]
     [ 4 19]
     [ 5 18]
     [ 6 16]
     [ 7 15]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAfold") -> None:
        super().__init__(path)

    @command(allowed_options=_ENERGY_OPTIONS + ["gquad", "maxBPspan", "circ"])
    def run(
        self,
        sequence: NucleotideSequence,
        temperature: float = 37,
        pairs: Any = None,
        paired: Any = None,
        unpaired: Any = None,
        downstream: Any = None,
        upstream: Any = None,
        enforce: bool = False,
        **kwargs: Any,
    ) -> CommandSetup[FoldedRNA]:
        """
        Compute the minimum free energy secondary structure.

        Constraints of known paired or unpaired bases can be added to the
        folding algorithm via `pairs`, `paired`, `unpaired`, `downstream`
        and `upstream`.

        Parameters
        ----------
        sequence : NucleotideSequence
            The RNA sequence.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        pairs : ndarray, shape=(n,2), dtype=int, optional
            Positions of constrained base pairs.
        paired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any other base.
        unpaired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are unpaired.
        downstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any downstream base.
        upstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any upstream base.
        enforce : bool, optional
            If set to true, the given constraints are enforced, i.e. the respective base
            pairs must form.
            By default, a constraint only forbids the formation of a base pair that
            would conflict with this constraint.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of FoldedRNA
            A handle to the running computation.
        """
        constraints = _build_constraints(
            len(sequence), pairs, paired, unpaired, downstream, upstream
        )
        # RNAfold reads the constraint from the line after the sequence
        stdin_text = f"{sequence}\n"
        if constraints is not None:
            stdin_text += f"{constraints}\n"

        parameters: list[CLIParameter] = [CLIFlag("noPS"), CLIOption("T", temperature)]
        if enforce:
            parameters.append(CLIFlag("enforceConstraint"))
        if constraints is not None:
            parameters.append(CLIFlag("C"))

        def evaluate(stdout: bytes, stderr: bytes) -> FoldedRNA:
            dot_bracket, free_energy = _parse_structure(_structure_lines(stdout)[-1])
            return FoldedRNA(sequence, dot_bracket, free_energy)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=stdin_text.encode(),
        )


class SuboptApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAsubopt`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAsubopt`` binary.

    Examples
    --------

    >>> results = SuboptApp().run(
    ...     NucleotideSequence("CCGGTGAAACGCTGG"), energy_range=2.0
    ... ).result()
    >>> print(len(results))
    3
    >>> for result in sorted(results, key=lambda result: result.free_energy):
    ...     print(result.dot_bracket, result.free_energy)
    ((((((...)))))) -5.3
    (((((.....))))) -3.9
    .(((((...))))). -3.5
    """

    def __init__(self, path: PathLike[str] | str = "RNAsubopt") -> None:
        super().__init__(path)

    @command(
        allowed_options=_ENERGY_OPTIONS
        + ["gquad", "maxBPspan", "circ", "sorted", "logML"]
    )
    def run(
        self,
        sequence: NucleotideSequence,
        energy_range: float = 1.0,
        temperature: float = 37,
        pairs: Any = None,
        paired: Any = None,
        unpaired: Any = None,
        downstream: Any = None,
        upstream: Any = None,
        enforce: bool = False,
        **kwargs: Any,
    ) -> CommandSetup[list[FoldedRNA]]:
        """
        Compute suboptimal secondary structures.

        Constraints of known paired or unpaired bases can be added to the
        folding algorithm via `pairs`, `paired`, `unpaired`, `downstream`
        and `upstream`.

        Parameters
        ----------
        sequence : NucleotideSequence
            The RNA sequence.
        energy_range : float, optional
            All structures with a free energy in this range (kcal/mol)
            above the minimum free energy are computed.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        pairs : ndarray, shape=(n,2), dtype=int, optional
            Positions of constrained base pairs.
        paired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any other base.
        unpaired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are unpaired.
        downstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any downstream base.
        upstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any upstream base.
        enforce : bool, optional
            If set to true, the given constraints are enforced, i.e. the respective base
            pairs must form.
            By default, a constraint only forbids the formation of a base pair that
            would conflict with this constraint.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of list of FoldedRNA
            A handle to the running computation.
        """
        constraints = _build_constraints(
            len(sequence), pairs, paired, unpaired, downstream, upstream
        )
        # RNAsubopt reads the constraint from the line after the sequence
        stdin_text = f"{sequence}\n"
        if constraints is not None:
            stdin_text += f"{constraints}\n"

        parameters: list[CLIParameter] = [
            CLIOption("e", energy_range),
            CLIOption("T", temperature),
        ]
        if enforce:
            parameters.append(CLIFlag("enforceConstraint"))
        if constraints is not None:
            parameters.append(CLIFlag("C"))

        def evaluate(stdout: bytes, stderr: bytes) -> list[FoldedRNA]:
            results = []
            for line in _structure_lines(stdout):
                dot_bracket, free_energy = _parse_structure(line)
                results.append(FoldedRNA(sequence, dot_bracket, free_energy))
            return results

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=stdin_text.encode(),
        )


class PKplexApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAPKplex`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAPKplex`` binary.

    Examples
    --------

    >>> sequence = NucleotideSequence("CCCCCGGGGGAAAAAGGGGGAAAAACCCCC")
    >>> result = PKplexApp().run(sequence).result()
    >>> print(result.dot_bracket)
    ((((([[[[[.....))))).....]]]]]
    >>> print(result.free_energy)
    -16.6
    >>> print(result.base_pairs())
    [[ 0 19]
     [ 1 18]
     [ 2 17]
     [ 3 16]
     [ 4 15]
     [ 5 29]
     [ 6 28]
     [ 7 27]
     [ 8 26]
     [ 9 25]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAPKplex") -> None:
        super().__init__(path)

    @command(
        allowed_options=["noLP", "noGU", "noClosingGU", "paramFile", "salt", "nsp"]
    )
    def run(
        self,
        sequence: NucleotideSequence,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[FoldedRNA]:
        """
        Predict a secondary structure including pseudoknots.

        Parameters
        ----------
        sequence : NucleotideSequence
            The RNA sequence.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of FoldedRNA
            A handle to the running computation.
            The ``dot_bracket`` may contain pseudoknot brackets
            (``[]``).
        """
        # RNAPKplex has no option to suppress the 'PKplex.ps' plot it writes
        # -> run in a dedicated directory to avoid polluting the CWD
        exec_dir = tempfile.TemporaryDirectory()
        parameters: list[CLIParameter] = [CLIOption("T", temperature)]

        def evaluate(stdout: bytes, stderr: bytes) -> FoldedRNA:
            dot_bracket, free_energy = _parse_structure(_structure_lines(stdout)[-1])
            return FoldedRNA(sequence, dot_bracket, free_energy)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=lambda: _cleanup(exec_dir),
            input=f"{sequence}\n".encode(),
            exec_dir=exec_dir.name,
        )


class LfoldApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNALfold`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNALfold`` binary.

    Examples
    --------

    >>> sequence = NucleotideSequence("GGGGAAAACCCCTTTTGGGGAAAACCCC")
    >>> results = LfoldApp().run(sequence, max_span=30).result()
    >>> for result in results:
    ...     print(result.dot_bracket, result.span, result.free_energy)
    .((.....)). (17, 28) -0.1
    .((((....)))) (15, 28) -5.4
    ((((....((((....))))....)))) (0, 28) -14.0
    >>> print(results[0].base_pairs())
    [[18 26]
     [19 25]]
    """

    def __init__(self, path: PathLike[str] | str = "RNALfold") -> None:
        super().__init__(path)

    @command(allowed_options=_ENERGY_OPTIONS + ["gquad"])
    def run(
        self,
        sequence: NucleotideSequence,
        max_span: int = 150,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[list[FoldedRNA]]:
        """
        Compute locally stable secondary structures.

        Parameters
        ----------
        sequence : NucleotideSequence
            The RNA sequence.
        max_span : int, optional
            The maximum allowed span (number of bases) of a base pair.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of list of FoldedRNA
            A handle to the running computation.
            Each structure describes a local part of `sequence`, given by
            its :attr:`FoldedRNA.span`.
        """
        parameters: list[CLIParameter] = [
            CLIOption("L", max_span),
            CLIOption("T", temperature),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> list[FoldedRNA]:
            results = []
            for line in _structure_lines(stdout):
                # RNALfold appends the global MFE as a trailing line without a
                # start position, which is not a local structure
                trailing = line.rsplit(")", 1)[1].split()
                if not trailing:
                    continue
                dot_bracket, free_energy = _parse_structure(line)
                # The trailing integer is the 1-based start position
                start = int(trailing[0]) - 1
                span = (start, start + len(dot_bracket))
                results.append(FoldedRNA(sequence, dot_bracket, free_energy, span))
            return results

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=f"{sequence}\n".encode(),
        )


class CofoldApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAcofold`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAcofold`` binary.

    Examples
    --------

    >>> result = CofoldApp().run(
    ...     NucleotideSequence("GGGGGGG"), NucleotideSequence("CCCCCCC")
    ... ).result()
    >>> print(result.dot_bracket)
    (((((((&)))))))
    >>> print(result.free_energy)
    -15.7
    >>> print(result.base_pairs())
    [[0 0 1 6]
     [0 1 1 5]
     [0 2 1 4]
     [0 3 1 3]
     [0 4 1 2]
     [0 5 1 1]
     [0 6 1 0]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAcofold") -> None:
        super().__init__(path)

    @command(allowed_options=_ENERGY_OPTIONS + ["gquad", "maxBPspan"])
    def run(
        self,
        sequence1: NucleotideSequence,
        sequence2: NucleotideSequence,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[CofoldedRNA]:
        """
        Compute the minimum free energy structure of a dimer.

        Parameters
        ----------
        sequence1, sequence2 : NucleotideSequence
            The two RNA strands.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of CofoldedRNA
            A handle to the running computation.
        """
        sequences = [sequence1, sequence2]
        parameters: list[CLIParameter] = [
            CLIFlag("noPS"),
            CLIOption("T", temperature),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> CofoldedRNA:
            dot_bracket, free_energy = _parse_structure(_structure_lines(stdout)[-1])
            return CofoldedRNA(sequences, dot_bracket, free_energy)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=_complex_string(sequences).encode(),
        )


class MultifoldApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAmultifold`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAmultifold`` binary.

    Examples
    --------

    >>> result = MultifoldApp().run([
    ...     NucleotideSequence("GGGGGGG"),
    ...     NucleotideSequence("CCCGGGG"),
    ...     NucleotideSequence("CCCCCCC"),
    ... ]).result()
    >>> print(result.dot_bracket)
    (((((((&)))(((.&)))))))
    >>> print(result.free_energy)
    -13.8
    >>> print(result.base_pairs())
    [[0 0 2 6]
     [0 1 2 5]
     [0 2 2 4]
     [0 3 2 3]
     [0 4 1 2]
     [0 5 1 1]
     [0 6 1 0]
     [1 3 2 2]
     [1 4 2 1]
     [1 5 2 0]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAmultifold") -> None:
        super().__init__(path)

    @command(allowed_options=_ENERGY_OPTIONS + ["gquad", "maxBPspan"])
    def run(
        self,
        sequences: Sequence[NucleotideSequence],
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[CofoldedRNA]:
        """
        Compute the minimum free energy structure of a complex.

        Parameters
        ----------
        sequences : sequence of NucleotideSequence
            The RNA strands forming the complex.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of CofoldedRNA
            A handle to the running computation.
        """
        sequences = list(sequences)
        parameters: list[CLIParameter] = [CLIOption("T", temperature)]

        def evaluate(stdout: bytes, stderr: bytes) -> CofoldedRNA:
            dot_bracket, free_energy = _parse_structure(_structure_lines(stdout)[-1])
            return CofoldedRNA(sequences, dot_bracket, free_energy)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=_complex_string(sequences).encode(),
        )


class DuplexApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAduplex`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAduplex`` binary.

    Examples
    --------

    >>> result = DuplexApp().run(
    ...     NucleotideSequence("AAAAGGGGGGG"), NucleotideSequence("CCCCCCCAAAA")
    ... ).result()
    >>> print(result.dot_bracket)
    .(((((((&))))))).
    >>> print(result.free_energy)
    -17.2
    >>> print(result.spans)
    [(3, 11), (0, 8)]
    >>> print(result.base_pairs())
    [[ 0  4  1  6]
     [ 0  5  1  5]
     [ 0  6  1  4]
     [ 0  7  1  3]
     [ 0  8  1  2]
     [ 0  9  1  1]
     [ 0 10  1  0]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAduplex") -> None:
        super().__init__(path)

    @command(
        allowed_options=["noLP", "noGU", "noClosingGU", "dangles", "paramFile", "salt"]
    )
    def run(
        self,
        sequence1: NucleotideSequence,
        sequence2: NucleotideSequence,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[CofoldedRNA]:
        """
        Predict the hybridization structure of two strands.

        Parameters
        ----------
        sequence1, sequence2 : NucleotideSequence
            The two RNA strands.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of CofoldedRNA
            A handle to the running computation.
            Only the interacting parts of the strands are covered by the
            structure, as given by :attr:`CofoldedRNA.spans`.
        """
        sequences = [sequence1, sequence2]
        parameters: list[CLIParameter] = [CLIOption("T", temperature)]

        def evaluate(stdout: bytes, stderr: bytes) -> CofoldedRNA:
            dot_bracket, spans, free_energy = _parse_duplex(
                _structure_lines(stdout)[-1]
            )
            return CofoldedRNA(sequences, dot_bracket, free_energy, spans)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=f"{sequence1}\n{sequence2}\n".encode(),
        )


class PlexApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAplex`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAplex`` binary.

    Examples
    --------

    >>> target = NucleotideSequence("AAAAAAAGGGGGGGGGGAAAAAAA")
    >>> query = NucleotideSequence("CCCCCCCCCC")
    >>> results = PlexApp().run(target, query).result()
    >>> print(results[0].dot_bracket)
    .((((((((((.&))))))))))
    >>> print(results[0].spans)
    [(6, 18), (0, 10)]
    >>> print(results[0].base_pairs())
    [[ 0  7  1  9]
     [ 0  8  1  8]
     [ 0  9  1  7]
     [ 0 10  1  6]
     [ 0 11  1  5]
     [ 0 12  1  4]
     [ 0 13  1  3]
     [ 0 14  1  2]
     [ 0 15  1  1]
     [ 0 16  1  0]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAplex") -> None:
        super().__init__(path)

    @command(
        allowed_options=[
            "paramFile",
            "salt",
            "interaction_length",
            "extension_cost",
            "energy_threshold",
            "duplex_distance",
        ]
    )
    def run(
        self,
        target: NucleotideSequence,
        query: NucleotideSequence,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[list[CofoldedRNA]]:
        """
        Find hybridization sites of a query in a target.

        Parameters
        ----------
        target : NucleotideSequence
            The (usually long) target RNA strand.
        query : NucleotideSequence
            The (usually short) query RNA strand.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of list of CofoldedRNA
            A handle to the running computation.
            In each :class:`CofoldedRNA` the first strand is the `target`
            and the second the `query`.
            Only the interacting parts are covered by the structure, as
            given by :attr:`CofoldedRNA.spans`.
        """
        sequences = [target, query]
        parameters: list[CLIParameter] = [CLIOption("T", temperature)]

        def evaluate(stdout: bytes, stderr: bytes) -> list[CofoldedRNA]:
            results = []
            for line in _structure_lines(stdout):
                if _RANGE_PATTERN.search(line) is None:
                    continue
                dot_bracket, spans, free_energy = _parse_duplex(line)
                results.append(CofoldedRNA(sequences, dot_bracket, free_energy, spans))
            return results

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=f"{target}\n{query}\n".encode(),
        )


class AlifoldApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAalifold`` software.

    In contrast to :class:`FoldApp`, the energy function includes a term
    that includes coevolution information extracted from an alignment in
    addition to the physical free energy term.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAalifold`` binary.

    Examples
    --------

    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> alignment = align_multiple(
    ...     [
    ...         NucleotideSequence("GGGGAAAACCCC"),
    ...         NucleotideSequence("GGGGAAAAACCCC"),
    ...         NucleotideSequence("GGGGGAAAACCCCC"),
    ...         NucleotideSequence("GGGAAAACCC"),
    ...     ],
    ...     matrix,
    ...     terminal_penalty=False,
    ... )[0]
    >>> print(alignment)
    -GGGGAAAACCCC-
    -GGGGAAAAACCCC
    GGGGGAAAACCCCC
    --GGGAAAACCC--
    >>> result = AlifoldApp().run(alignment).result()
    >>> print(result.dot_bracket)
    .((((....)))).
    >>> print(result.free_energy, result.covariance_energy)
    -4.47 0.31
    >>> print(result.base_pairs())
    [[ 1 12]
     [ 2 11]
     [ 3 10]
     [ 4  9]]
    >>> print(result.base_pairs_of(0))
    [[ 0 11]
     [ 1 10]
     [ 2  9]
     [ 3  8]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAalifold") -> None:
        super().__init__(path)

    @command(
        allowed_options=_ENERGY_OPTIONS
        + ["gquad", "maxBPspan", "circ", "cfactor", "nfactor", "old", "endgaps"]
    )
    def run(
        self,
        alignment: Alignment,
        temperature: float = 37,
        pairs: Any = None,
        paired: Any = None,
        unpaired: Any = None,
        downstream: Any = None,
        upstream: Any = None,
        enforce: bool = False,
        **kwargs: Any,
    ) -> CommandSetup[FoldedAlignment]:
        """
        Predict the consensus secondary structure of an alignment.

        Constraints of known paired or unpaired bases can be added to the
        folding algorithm via `pairs`, `paired`, `unpaired`, `downstream`
        and `upstream`.

        Parameters
        ----------
        alignment : Alignment
            An alignment of RNA sequences.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        pairs : ndarray, shape=(n,2), dtype=int, optional
            Positions of constrained base pairs.
        paired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any other base.
        unpaired : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are unpaired.
        downstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any downstream base.
        upstream : ndarray, shape=(n,), dtype=int or dtype=bool, optional
            Positions of bases that are paired with any upstream base.
        enforce : bool, optional
            If set to true, the given constraints are enforced, i.e. the respective base
            pairs must form.
            By default, a constraint only forbids the formation of a base pair that
            would conflict with this constraint.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of FoldedAlignment
            A handle to the running computation.
        """
        constraints = _build_constraints(
            len(alignment), pairs, paired, unpaired, downstream, upstream
        )
        in_file = _alignment_fasta(alignment)

        parameters: list[CLIParameter] = [CLIFlag("noPS"), CLIOption("T", temperature)]
        if enforce:
            parameters.append(CLIFlag("enforceConstraint"))

        constraint_input = None
        if constraints is not None:
            parameters.append(CLIFlag("C"))
            # The constraints are passed via STDIN
            constraint_input = constraints.encode()
        parameters.append(CLIArgument(in_file.name))

        def evaluate(stdout: bytes, stderr: bytes) -> FoldedAlignment:
            line = _structure_lines(stdout)[-1]
            dot_bracket = line.split()[0]
            free_energy, covariance_energy = _parse_alignment_energies(line)
            return FoldedAlignment(
                alignment, dot_bracket, free_energy, covariance_energy
            )

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=lambda: _cleanup(in_file),
            input=constraint_input,
        )


class LalifoldApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNALalifold`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNALalifold`` binary.

    Examples
    --------

    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> alignment = align_multiple(
    ...     [
    ...         NucleotideSequence("GGGGAAAACCCC"),
    ...         NucleotideSequence("GGGGAAAAACCCC"),
    ...         NucleotideSequence("GGGGGAAAACCCCC"),
    ...         NucleotideSequence("GGGAAAACCC"),
    ...     ],
    ...     matrix,
    ...     terminal_penalty=False,
    ... )[0]
    >>> print(alignment)
    -GGGGAAAACCCC-
    -GGGGAAAAACCCC
    GGGGGAAAACCCCC
    --GGGAAAACCC--
    >>> results = LalifoldApp().run(alignment, max_span=30).result()
    >>> for result in results:
    ...     print(result.dot_bracket, result.span, result.free_energy)
    .(((.....))). (1, 14) -2.2
    .((((....)))). (0, 14) -4.47
    >>> print(results[0].base_pairs())
    [[ 2 12]
     [ 3 11]
     [ 4 10]]
    >>> print(results[0].base_pairs_of(0))
    [[ 1 11]
     [ 2 10]
     [ 3  9]]
    """

    def __init__(self, path: PathLike[str] | str = "RNALalifold") -> None:
        super().__init__(path)

    @command(
        allowed_options=_ENERGY_OPTIONS + ["gquad", "cfactor", "nfactor", "threshold"]
    )
    def run(
        self,
        alignment: Alignment,
        max_span: int = 70,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[list[FoldedAlignment]]:
        """
        Compute locally stable consensus secondary structures.

        Parameters
        ----------
        alignment : Alignment
            An alignment of RNA sequences.
        max_span : int, optional
            The maximum allowed span (number of columns) of a base pair.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of list of FoldedAlignment
            A handle to the running computation.
            Each structure describes a local part of the alignment, given
            by its :attr:`FoldedAlignment.span`.
        """
        in_file = _alignment_fasta(alignment)
        parameters: list[CLIParameter] = [
            # Split the energy into physical and covariance contribution
            CLIFlag("split-contributions"),
            CLIOption("L", max_span),
            CLIOption("T", temperature),
            CLIArgument(in_file.name),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> list[FoldedAlignment]:
            results = []
            for line in _structure_lines(stdout):
                dot_bracket = line.split()[0]
                free_energy, covariance_energy = _parse_alignment_energies(line)
                # The trailing 'start - stop' is a 1-based inclusive range
                after = line.rsplit(")", 1)[1]
                start, stop = (int(value) for value in re.findall(r"\d+", after))
                span = (start - 1, stop)
                results.append(
                    FoldedAlignment(
                        alignment, dot_bracket, free_energy, covariance_energy, span
                    )
                )
            return results

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=lambda: _cleanup(in_file),
        )


class AliduplexApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAaliduplex`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAaliduplex`` binary.

    Examples
    --------

    >>> matrix = SubstitutionMatrix.std_nucleotide_matrix()
    >>> alignment1 = align_multiple(
    ...     [
    ...         NucleotideSequence("AAAAGGGGGGG"),
    ...         NucleotideSequence("AAACGGGGGGG"),
    ...         NucleotideSequence("AAAAAGGGGGGG"),
    ...         NucleotideSequence("AACGGGGGGG"),
    ...     ],
    ...     matrix,
    ...     terminal_penalty=False,
    ... )[0]
    >>> print(alignment1)
    -AAAAGGGGGGG
    -AAACGGGGGGG
    AAAAAGGGGGGG
    --AACGGGGGGG
    >>> alignment2 = align_multiple(
    ...     [
    ...         NucleotideSequence("CCCCCCCAAAA"),
    ...         NucleotideSequence("CCCCCCGAAAA"),
    ...         NucleotideSequence("CCCCCCCAAA"),
    ...         NucleotideSequence("CCCCCCCAAAAA"),
    ...     ],
    ...     matrix,
    ...     terminal_penalty=False,
    ... )[0]
    >>> print(alignment2)
    CCCCCCCAAAA-
    CCCCCCGAAAA-
    CCCCCCCAAA--
    CCCCCCCAAAAA
    >>> result = AliduplexApp().run(alignment1, alignment2).result()
    >>> print(result.dot_bracket)
    .(((((((&))))))).
    >>> print(result.free_energy)
    -16.17
    >>> print(result.base_pairs())
    [[ 0  5  1  6]
     [ 0  6  1  5]
     [ 0  7  1  4]
     [ 0  8  1  3]
     [ 0  9  1  2]
     [ 0 10  1  1]
     [ 0 11  1  0]]
    >>> print(result.base_pairs_of(0))
    [[ 0  4  1  6]
     [ 0  5  1  5]
     [ 0  6  1  4]
     [ 0  7  1  3]
     [ 0  8  1  2]
     [ 0  9  1  1]
     [ 0 10  1  0]]
    """

    def __init__(self, path: PathLike[str] | str = "RNAaliduplex") -> None:
        super().__init__(path)

    @command(
        allowed_options=["noLP", "noGU", "noClosingGU", "dangles", "paramFile", "salt"]
    )
    def run(
        self,
        alignment1: Alignment,
        alignment2: Alignment,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[CofoldedAlignment]:
        """
        Predict the hybridization structure of two alignments.

        Parameters
        ----------
        alignment1, alignment2 : Alignment
            The two RNA alignments.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of CofoldedAlignment
            A handle to the running computation.

        Notes
        -----
        Unlike :class:`AlifoldApp`, ``RNAaliduplex`` does not report the
        covariance contribution separately.
        Therefore :attr:`CofoldedAlignment.free_energy` contains the
        *total* energy, i.e. the sum of the physical free energy and the
        covariance term, and
        :attr:`CofoldedAlignment.covariance_energy` is ``None``.
        """
        alignments = [alignment1, alignment2]
        in_file_1 = _alignment_clustal(alignment1)
        in_file_2 = _alignment_clustal(alignment2)
        parameters: list[CLIParameter] = [
            CLIOption("T", temperature),
            CLIArgument(in_file_1.name),
            CLIArgument(in_file_2.name),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> CofoldedAlignment:
            dot_bracket, spans, free_energy = _parse_duplex(
                _structure_lines(stdout)[-1]
            )
            return CofoldedAlignment(alignments, dot_bracket, free_energy, None, spans)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            cleanup=lambda: _cleanup(in_file_1, in_file_2),
        )


class EvalApp(LocalApp):
    """
    A handle to *ViennaRNA's* ``RNAeval`` software.

    Parameters
    ----------
    path : str, optional
        Path of the ``RNAeval`` binary.

    Notes
    -----
    ``RNAeval`` ignores base pairs inside pseudoknots.

    Examples
    --------
    Evaluate the free energy of a secondary structure taken from a 3D structure:

    >>> pdbx_file = BinaryCIFFile.read(path_to_structures / "4p5j.bcif")
    >>> atoms = get_structure(pdbx_file, model=1)
    >>> nucleotide = atoms[filter_nucleotides(atoms)]
    >>> sequence = to_sequence(nucleotide)[0][0]
    >>> atom_pairs = base_pairs(nucleotide)
    >>> residue_pairs = get_residue_positions(nucleotide, atom_pairs.flatten()).reshape(-1, 2)
    >>> # 'from_base_pairs()' drops non-canonical pairs that RNAeval cannot score
    >>> folded = FoldedRNA.from_base_pairs(sequence, residue_pairs)
    >>> result = EvalApp().run(folded).result()
    >>> print(result.free_energy)
    -23.6
    """

    def __init__(self, path: PathLike[str] | str = "RNAeval") -> None:
        super().__init__(path)

    @command(allowed_options=["dangles", "paramFile", "salt", "nsp", "gquad", "circ"])
    def run(
        self,
        folded: FoldedRNA | CofoldedRNA,
        temperature: float = 37,
        **kwargs: Any,
    ) -> CommandSetup[FoldedRNA | CofoldedRNA]:
        """
        Evaluate the free energy of a secondary structure.

        Both single strands and complexes of multiple strands are
        supported.

        Parameters
        ----------
        folded : FoldedRNA or CofoldedRNA
            The RNA sequence(s) and secondary structure to evaluate.
            Its ``free_energy`` is ignored and may be ``None``.
        temperature : float, optional
            The temperature (°C) to be assumed for the energy parameters.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of FoldedRNA or CofoldedRNA
            A handle to the running computation.
            The returned structure equals the input, with its evaluated
            free energy attached, and is of the same type as `folded`.
        """
        # RNAeval cannot score pseudoknots: the '[]{}<>' brackets are parsed
        # but ignored, and the letters for higher-order pseudoknots even make it error
        dot_bracket = folded.dot_bracket
        pseudoknot_chars = set(dot_bracket) - _NESTED_STRUCTURE_CHARS
        if pseudoknot_chars:
            warnings.warn(
                "The structure contains pseudoknots, which RNAeval cannot "
                "score; their base pairs are treated as unpaired",
                UserWarning,
            )
            for char in pseudoknot_chars:
                dot_bracket = dot_bracket.replace(char, ".")
        # RNAeval requires the structure to span the full sequence length,
        # so a local structure is padded with unpaired positions
        if isinstance(folded, CofoldedRNA):
            sequence_line = "&".join(str(sequence) for sequence in folded.sequences)
            spans = (
                folded.spans
                if folded.spans is not None
                else [None] * len(folded.sequences)
            )
            structure_line = "&".join(
                _pad_structure(segment, len(sequence), span)
                for segment, sequence, span in zip(
                    dot_bracket.split("&"), folded.sequences, spans
                )
            )
        else:
            sequence_line = str(folded.sequence)
            structure_line = _pad_structure(
                dot_bracket, len(folded.sequence), folded.span
            )
        parameters: list[CLIParameter] = [CLIOption("T", temperature)]

        def evaluate(stdout: bytes, stderr: bytes) -> FoldedRNA | CofoldedRNA:
            _, free_energy = _parse_structure(_structure_lines(stdout)[-1])
            if isinstance(folded, CofoldedRNA):
                return CofoldedRNA(
                    folded.sequences, dot_bracket, free_energy, folded.spans
                )
            return FoldedRNA(folded.sequence, dot_bracket, free_energy, folded.span)

        return CommandSetup(
            parameters=parameters,
            evaluate=evaluate,
            input=f"{sequence_line}\n{structure_line}\n".encode(),
        )


def _floats(text: str) -> list[float]:
    """
    Extract all floating point numbers (i.e. numbers with a decimal
    point) from a string.
    Integers such as position indices are ignored.
    """
    return [float(match) for match in _FLOAT_PATTERN.findall(text)]


def _is_structure_token(token: str) -> bool:
    """
    Check whether a whitespace-separated token is a dot-bracket
    structure, i.e. consists exclusively of dot-bracket characters and
    the strand separator ``&``.
    """
    return len(token) > 0 and all(char in _STRUCTURE_TOKEN_CHARS for char in token)


def _structure_lines(stdout: bytes) -> list[str]:
    """
    Return all lines of the output whose first whitespace-separated token
    is a dot-bracket structure.
    """
    lines = []
    for line in stdout.decode("UTF-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _is_structure_token(stripped.split()[0]):
            lines.append(stripped)
    return lines


def _pad_structure(segment: str, length: int, span: tuple[int, int] | None) -> str:
    """
    Pad a local dot-bracket segment with unpaired positions, so it spans
    the full sequence `length`.
    """
    if span is None:
        return segment
    start, stop = span
    return "." * start + segment + "." * (length - stop)


def _parse_structure(line: str) -> tuple[str, float]:
    """
    Parse a ``dot-bracket [(]energy[)]`` line into structure and energy.
    """
    dot_bracket = line.split()[0]
    return dot_bracket, _floats(line)[0]


def _parse_duplex(
    line: str,
) -> tuple[str, list[tuple[int, int]], float]:
    """
    Parse a duplex output line of the form
    ``structure  from,to : from,to  (energy)`` into the structure, the
    per-strand spans (0-based, exclusive stop) and the energy.
    """
    dot_bracket = line.split()[0]
    match = _RANGE_PATTERN.search(line)
    if match is None:
        raise ValueError(f"Could not parse strand ranges from '{line}'")
    from_1, to_1, from_2, to_2 = (int(group) for group in match.groups())
    spans = [(from_1 - 1, to_1), (from_2 - 1, to_2)]
    # Ranges contain no decimal point, so the first float is the energy
    return dot_bracket, spans, _floats(line)[0]


def _parse_alignment_energies(line: str) -> tuple[float, float | None]:
    """
    Parse the energy part of an alignment-based output line into the
    physical free energy and the covariance energy.

    The covariance energy is only available if the line contains the
    decomposition ``(total = free + covariance)``.
    """
    energies = _floats(line)
    if len(energies) >= 3:
        # (total = free + covariance)
        return energies[1], energies[2]
    return energies[0], None


def _complex_string(sequences: Sequence[NucleotideSequence]) -> str:
    """
    Join multiple strands into a single ``&``-separated input line.
    """
    return "&".join(str(sequence) for sequence in sequences) + "\n"


def _alignment_fasta(alignment: Alignment) -> Any:
    """
    Write an alignment into a temporary aligned FASTA file.
    """
    file = NamedTemporaryFile("w", suffix=".fa", delete=False)
    fasta_file = FastaFile(chars_per_line=np.iinfo(np.int32).max)
    set_alignment(
        fasta_file,
        alignment,
        seq_names=[str(i) for i in range(len(alignment.sequences))],
    )
    fasta_file.write(file)
    file.flush()
    return file


def _alignment_clustal(alignment: Alignment) -> Any:
    """
    Write an alignment into a temporary CLUSTAL file.
    """
    file = NamedTemporaryFile("w", suffix=".aln", delete=False)
    clustal_file = ClustalFile()
    set_clustal_alignment(
        clustal_file,
        alignment,
        seq_names=[str(i) for i in range(len(alignment.sequences))],
    )
    clustal_file.write(file)
    file.flush()
    return file


def _build_constraints(
    length: int,
    pairs: Any,
    paired: Any,
    unpaired: Any,
    downstream: Any,
    upstream: Any,
) -> str | None:
    if all(
        constraint is None
        for constraint in (pairs, paired, unpaired, downstream, upstream)
    ):
        return None
    return build_constraint_string(
        length, pairs, paired, unpaired, downstream, upstream
    )


def _cleanup(*resources: Any) -> None:
    for resource in resources:
        if resource is None:
            continue
        if isinstance(resource, tempfile.TemporaryDirectory):
            resource.cleanup()
        else:
            cleanup_tempfile(resource)
