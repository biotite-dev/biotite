# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.sra"
__author__ = "Patrick Kunzmann"
__all__ = [
    "PrefetchApp",
    "PrefetchDirectory",
    "FastqDumpApp",
    "FastqResult",
    "FastaResult",
]

import glob
from dataclasses import dataclass, field
from os import PathLike
from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, TypeAlias
import numpy as np
from biotite.application_v2.localapp import (
    CLIArgument,
    CLIFlag,
    CLIOption,
    CLIParameter,
    CommandSetup,
    LocalApp,
    command,
)
from biotite.sequence.io.fasta.convert import get_sequences as _fasta_get_sequences
from biotite.sequence.io.fasta.file import FastaFile
from biotite.sequence.io.fastq.convert import get_sequences as _fastq_get_sequences
from biotite.sequence.io.fastq.file import FastqFile
from biotite.sequence.seqtypes import NucleotideSequence

_OffsetFormat: TypeAlias = Literal[
    "Sanger", "Solexa", "Illumina-1.3", "Illumina-1.5", "Illumina-1.8"
]


@dataclass(frozen=True)
class PrefetchDirectory:
    """
    A handle to the directory an SRA run was prefetched into.

    It can be passed to a dump application to avoid downloading the run
    again.

    Attributes
    ----------
    directory : Path
        The path to the ``prefetch`` output directory.
    """

    directory: Path
    # Keep a temporary directory alive as long as this handle exists,
    # so the prefetched data is available until the handle is discarded
    _temp_dir: TemporaryDirectory[str] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True)
class FastqResult:
    """
    The result of a FASTQ extraction with :meth:`FastqDumpApp.extract_fastq`.

    The parsed files and sequences are not kept in memory, but are read
    from :attr:`file_paths` on demand via :meth:`get_files`,
    :meth:`get_sequences` and :meth:`get_sequences_and_scores`.

    Attributes
    ----------
    file_paths : list of Path
        The paths to the extracted FASTQ files.
        There is one file per read in a spot.
    """

    file_paths: list[Path]
    _temp_dir: TemporaryDirectory[str] | None = field(
        default=None, repr=False, compare=False
    )

    def get_files(self, offset: int | _OffsetFormat = "Sanger") -> list[FastqFile]:
        """
        Parse the extracted FASTQ files.

        Parameters
        ----------
        offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
            This value is subtracted from the FASTQ ASCII code to obtain
            the quality score.

        Returns
        -------
        files : list of FastqFile
            The parsed FASTQ files, one per read in a spot.
        """
        return [FastqFile.read(path, offset=offset) for path in self.file_paths]

    def get_sequences(
        self, offset: int | _OffsetFormat = "Sanger"
    ) -> list[dict[str, NucleotideSequence]]:
        """
        Get the reads from the extracted FASTQ files.

        Parameters
        ----------
        offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
            This value is subtracted from the FASTQ ASCII code to obtain
            the quality score.

        Returns
        -------
        sequences : list of dict (str -> NucleotideSequence)
            The reads for each spot: the first item contains the first
            read for each spot, the second item the second read, etc.
            Each item maps identifiers to their sequence.
        """
        return [
            {header: sequence for header, (sequence, _) in spot.items()}
            for spot in self.get_sequences_and_scores(offset)
        ]

    def get_sequences_and_scores(
        self, offset: int | _OffsetFormat = "Sanger"
    ) -> list[dict[str, tuple[NucleotideSequence, np.ndarray]]]:
        """
        Get the reads and quality scores from the extracted FASTQ files.

        Parameters
        ----------
        offset : int or {'Sanger', 'Solexa', 'Illumina-1.3', 'Illumina-1.5', 'Illumina-1.8'}, optional
            This value is subtracted from the FASTQ ASCII code to obtain
            the quality score.

        Returns
        -------
        sequences_and_scores : list of dict (str -> tuple(NucleotideSequence, ndarray))
            Like the result of :meth:`get_sequences`, but each value
            additionally contains the quality scores.
        """
        return [_fastq_get_sequences(file) for file in self.get_files(offset)]


@dataclass(frozen=True)
class FastaResult:
    """
    The result of a FASTA extraction with :meth:`FastqDumpApp.extract_fasta`.

    The parsed files and sequences are not kept in memory, but are read
    from :attr:`file_paths` on demand via :meth:`get_files` and
    :meth:`get_sequences`.

    Attributes
    ----------
    file_paths : list of Path
        The paths to the extracted FASTA files.
        There is one file per read in a spot.
    """

    file_paths: list[Path]
    _temp_dir: TemporaryDirectory[str] | None = field(
        default=None, repr=False, compare=False
    )

    def get_files(self) -> list[FastaFile]:
        """
        Parse the extracted FASTA files.

        Returns
        -------
        files : list of FastaFile
            The parsed FASTA files, one per read in a spot.
        """
        return [FastaFile.read(path) for path in self.file_paths]

    def get_sequences(self) -> list[dict[str, NucleotideSequence]]:
        """
        Get the reads from the extracted FASTA files.

        Returns
        -------
        sequences : list of dict (str -> NucleotideSequence)
            The reads for each spot: the first item contains the first
            read for each spot, the second item the second read, etc.
            Each item maps identifiers to their sequence.
        """
        return [
            _fasta_get_sequences(file, seq_type=NucleotideSequence)
            for file in self.get_files()
        ]


class PrefetchApp(LocalApp):
    """
    A handle to ``prefetch`` from *sra-tools*.

    Parameters
    ----------
    path : str, optional
        Path of the ``prefetch`` binary.

    Notes
    -----
    According to the *sra-tools*
    `documentation <https://github.com/ncbi/sra-tools/wiki/08.-prefetch-and-fasterq-dump>`_,
    running ``prefetch`` before ``fastq-dump``
    '*is the fastest way to extract FASTQ-files from SRA-accessions*'.

    Examples
    --------

    >>> prefetch = PrefetchApp().run("ERR11344941").result()  # doctest: +SKIP
    >>> sequences = FastqDumpApp().extract_fastq(  # doctest: +SKIP
    ...     "ERR11344941", prefetch=prefetch
    ... ).result().get_sequences()
    """

    def __init__(self, path: PathLike[str] | str = "prefetch") -> None:
        super().__init__(path)

    @command(
        allowed_options=["transport", "min_size", "max_size", "verify", "resume", "ngc"]
    )
    def run(
        self,
        accession: str,
        output_directory: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[PrefetchDirectory]:
        """
        Download the run with the given accession.

        Parameters
        ----------
        accession : str
            The accession of the run to be downloaded.
        output_directory : str or Path, optional
            The directory the run is downloaded into.
            By default, a temporary directory is used, that is deleted
            once the returned :class:`PrefetchDirectory` is discarded.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of PrefetchDirectory
            A handle to the running download.
            Call :meth:`Future.result()` to obtain the
            :class:`PrefetchDirectory`.
        """
        if output_directory is None:
            temp_dir = TemporaryDirectory(suffix="_sra")
            directory = Path(temp_dir.name)
        else:
            temp_dir = None
            directory = Path(output_directory)

        parameters: list[CLIParameter] = [
            CLIFlag("q"),
            CLIOption("O", directory),
            CLIArgument(accession),
        ]

        def evaluate(stdout: bytes, stderr: bytes) -> PrefetchDirectory:
            return PrefetchDirectory(directory=directory, _temp_dir=temp_dir)

        return CommandSetup(parameters=parameters, evaluate=evaluate)


class FastqDumpApp(LocalApp):
    """
    A handle to ``fasterq-dump`` from *sra-tools*.

    The reads can be extracted as FASTQ, i.e. including quality scores,
    via :meth:`extract_fastq` or as FASTA via :meth:`extract_fasta`.

    Parameters
    ----------
    path : str, optional
        Path of the ``fasterq-dump`` binary.

    Notes
    -----
    According to the *sra-tools*
    `documentation <https://github.com/ncbi/sra-tools/wiki/08.-prefetch-and-fasterq-dump>`_,
    running ``prefetch`` before ``fastq-dump``
    '*is the fastest way to extract FASTQ-files from SRA-accessions*'.

    Examples
    --------

    >>> app = FastqDumpApp()
    >>> sequences = app.extract_fastq("ERR11344941").result().get_sequences()  # doctest: +SKIP
    """

    def __init__(self, path: PathLike[str] | str = "fasterq-dump") -> None:
        super().__init__(path)

    @command(
        allowed_options=[
            "threads",
            "mem",
            "temp",
            "bufsize",
            "curcache",
            "skip_technical",
            "include_technical",
            "min_read_len",
            "bases",
        ]
    )
    def extract_fastq(
        self,
        accession: str,
        prefetch: PrefetchDirectory | None = None,
        output_path_prefix: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[FastqResult]:
        """
        Extract the run with the given accession as FASTQ.

        Parameters
        ----------
        accession : str
            The accession of the run to be extracted.
        prefetch : PrefetchDirectory, optional
            A directory a previous :class:`PrefetchApp` downloaded the run
            into.
            If not given, the run is downloaded on the fly.
        output_path_prefix : str or Path, optional
            The prefix of the path to store the extracted FASTQ file(s).
            ``.fastq`` is appended for a single read per spot, ``_1.fastq``,
            ``_2.fastq``, etc. for multiple reads per spot.
            By default, a temporary directory is used.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of FastqResult
            A handle to the running extraction.
            Call :meth:`Future.result()` to obtain the
            :class:`FastqResult`.
        """
        parameters, temp_dir, prefix, suffix = _setup_parameters(
            accession, prefetch, output_path_prefix, fasta=False
        )

        def evaluate(stdout: bytes, stderr: bytes) -> FastqResult:
            return FastqResult(
                file_paths=_find_files(prefix, suffix), _temp_dir=temp_dir
            )

        return CommandSetup(parameters=parameters, evaluate=evaluate)

    @command(
        allowed_options=[
            "threads",
            "mem",
            "temp",
            "bufsize",
            "curcache",
            "skip_technical",
            "include_technical",
            "min_read_len",
            "bases",
        ]
    )
    def extract_fasta(
        self,
        accession: str,
        prefetch: PrefetchDirectory | None = None,
        output_path_prefix: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[FastaResult]:
        """
        Extract the run with the given accession as FASTA.

        Parameters
        ----------
        accession : str
            The accession of the run to be extracted.
        prefetch : PrefetchDirectory, optional
            A directory a previous :class:`PrefetchApp` downloaded the run
            into.
            If not given, the run is downloaded on the fly.
        output_path_prefix : str or Path, optional
            The prefix of the path to store the extracted FASTA file(s).
            By default, a temporary directory is used.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of FastaResult
            A handle to the running extraction.
            Call :meth:`Future.result()` to obtain the
            :class:`FastaResult`.
        """
        parameters, temp_dir, prefix, suffix = _setup_parameters(
            accession, prefetch, output_path_prefix, fasta=True
        )

        def evaluate(stdout: bytes, stderr: bytes) -> FastaResult:
            return FastaResult(
                file_paths=_find_files(prefix, suffix), _temp_dir=temp_dir
            )

        return CommandSetup(parameters=parameters, evaluate=evaluate)


def _setup_parameters(
    accession: str,
    prefetch: PrefetchDirectory | None,
    output_path_prefix: PathLike[str] | str | None,
    fasta: bool,
) -> tuple[list[CLIParameter], TemporaryDirectory[str] | None, str, str]:
    """
    Assemble the ``fasterq-dump`` command line parameters for a run.
    """
    if output_path_prefix is None:
        temp_dir = TemporaryDirectory(suffix="_sra")
        prefix = join(temp_dir.name, accession)
    else:
        temp_dir = None
        prefix = str(output_path_prefix)
    # Without a prefetched directory, the accession is downloaded on the fly
    input_path = accession if prefetch is None else prefetch.directory / accession

    # 'fasterq-dump' uses the given output file name verbatim,
    # so the extension must match the actual output format
    suffix = ".fasta" if fasta else ".fastq"
    parameters: list[CLIParameter] = [
        CLIFlag("q"),
        CLIOption("o", prefix + suffix),
    ]
    if fasta:
        parameters.append(CLIFlag("fasta"))
    parameters.append(CLIArgument(input_path))
    return parameters, temp_dir, prefix, suffix


def _find_files(prefix: str, suffix: str) -> list[Path]:
    """
    Find the FASTQ/FASTA files written by ``fasterq-dump`` for a run.
    """
    return [
        Path(path)
        for path in sorted(
            # Entries with one read per spot
            glob.glob(prefix + suffix)
            # Entries with multiple reads per spot
            + glob.glob(prefix + "_*" + suffix)
        )
    ]
