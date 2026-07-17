# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from __future__ import annotations

__name__ = "biotite.application_v2.mmseqs"
__author__ = "Patrick Kunzmann"
__all__ = [
    "DatabaseType",
    "AlignmentFormatMode",
    "MSAFormatMode",
    "ProfileMatchMode",
    "VerbosityLevel",
    "SearchType",
    "AlignmentMode",
    "CoverageMode",
    "ClusterMode",
    "SequenceIdentityMode",
    "StructureAlignmentType",
    "UnpackNameMode",
    "MMseqsApp",
    "FoldseekApp",
]

from collections.abc import Iterable
from enum import IntEnum
from io import IOBase
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir
from typing import IO, Any, ClassVar, overload
import numpy as np
from biotite.application_v2.localapp import (
    CLIArgument,
    CLIOption,
    CommandSetup,
    LocalApp,
    LocalProcessFuture,
    command,
)
from biotite.application_v2.mmseqs.database import (
    AlignmentDatabase,
    ClusterDatabase,
    Database,
    MSADatabase,
    ProfileDatabase,
    SequenceDatabase,
)
from biotite.application_v2.msa import resolve_gap_penalty
from biotite.sequence.align.matrix import SubstitutionMatrix
from biotite.sequence.seqtypes import ProteinSequence


class DatabaseType(IntEnum):
    """
    Sequence types accepted by the MMseqs2 ``createdb`` command.

    Attributes
    ----------
    AUTO
        Detect the type from the input sequences.
    AMINO_ACID
        Treat the input as protein sequences.
    NUCLEOTIDE
        Treat the input as nucleotide sequences.
    """

    AUTO = 0
    AMINO_ACID = 1
    NUCLEOTIDE = 2


class AlignmentFormatMode(IntEnum):
    """
    Output modes accepted by the ``convertalis`` command.

    Attributes
    ----------
    BLAST_TABLE
        A *BLAST*-like tabular format.
    SAM
        The *SAM* alignment format.
    BLAST_TABLE_WITH_LENGTH
        Like :attr:`BLAST_TABLE`, with the query and target length appended.
    HTML
        A rendered HTML report.
    BLAST_TABLE_WITH_HEADERS
        Like :attr:`BLAST_TABLE`, preceded by a row of column names.
    CALPHA_PDB
        The superimposed C-alpha coordinates as PDB, only in *Foldseek*.

    Notes
    -----
    Only :attr:`BLAST_TABLE` and :attr:`BLAST_TABLE_WITH_HEADERS` support
    selecting the reported columns via the ``format_output`` option.
    """

    BLAST_TABLE = 0
    SAM = 1
    BLAST_TABLE_WITH_LENGTH = 2
    HTML = 3
    BLAST_TABLE_WITH_HEADERS = 4
    CALPHA_PDB = 5


class MSAFormatMode(IntEnum):
    """
    Output modes accepted by the ``result2msa`` command.

    Attributes
    ----------
    COMPACT_A3M
        The binary *cA3M* format.
    COMPACT_A3M_WITH_CONSENSUS
        The binary *cA3M* format, including the consensus sequence.
    FASTA
        Aligned FASTA, i.e. all sequences are padded with gaps to equal length.
    FASTA_WITH_INFO
        Aligned FASTA, preceded by a summary header.
    STOCKHOLM
        A *Stockholm* flat file instead of a database.
    A3M
        The *A3M* format, i.e. insertions into the query appear as lower case
        letters instead of as gaps in the other sequences.
    A3M_WITH_INFO
        The *A3M* format, including the alignment information.
    """

    COMPACT_A3M = 0
    COMPACT_A3M_WITH_CONSENSUS = 1
    FASTA = 2
    FASTA_WITH_INFO = 3
    STOCKHOLM = 4
    A3M = 5
    A3M_WITH_INFO = 6


class ProfileMatchMode(IntEnum):
    """
    Column selection modes accepted by the ``msa2profile`` command.

    Attributes
    ----------
    FIRST_SEQUENCE
        Keep the columns that have a residue in the first sequence of the MSA.
        As the first sequence is the query, the profile is kept in the
        coordinate frame of the query sequence.
    MATCH_RATIO
        Keep the columns that have a residue in at least ``match_ratio`` of all
        sequences in the MSA.
    """

    FIRST_SEQUENCE = 0
    MATCH_RATIO = 1


class VerbosityLevel(IntEnum):
    """
    Values accepted by the ``v`` option of every command.

    Each level reports the messages of the preceding ones as well.

    Attributes
    ----------
    QUIET
        Report nothing.
    ERRORS
        Report errors.
    WARNINGS
        Also report warnings.
    INFO
        Also report progress information.
    """

    QUIET = 0
    ERRORS = 1
    WARNINGS = 2
    INFO = 3


class SearchType(IntEnum):
    """
    Values accepted by the ``search_type`` option.

    Attributes
    ----------
    AUTO
        Detect the type from the database.
    AMINO_ACID
        Align protein sequences to protein sequences.
    TRANSLATED
        Translate the nucleotide sequences and align the products.
    NUCLEOTIDE
        Align nucleotide sequences to nucleotide sequences.
    TRANSLATED_NUCLEOTIDE
        Align translated queries to nucleotide targets.
    """

    AUTO = 0
    AMINO_ACID = 1
    TRANSLATED = 2
    NUCLEOTIDE = 3
    TRANSLATED_NUCLEOTIDE = 4


class AlignmentMode(IntEnum):
    """
    Values accepted by the ``alignment_mode`` option.

    Each mode computes the quantities of the preceding one as well.

    Attributes
    ----------
    AUTO
        Let the application choose based on the other options.
    SCORE_AND_END
        Compute only the alignment score and the end position.
    START_AND_COVERAGE
        Also compute the start position and the coverage.
    SEQUENCE_IDENTITY
        Also compute the sequence identity.
    UNGAPPED
        Compute only an ungapped alignment.
    """

    AUTO = 0
    SCORE_AND_END = 1
    START_AND_COVERAGE = 2
    SEQUENCE_IDENTITY = 3
    UNGAPPED = 4


class CoverageMode(IntEnum):
    """
    Values accepted by the ``cov_mode`` option.

    The threshold itself is given by the ``c`` option.

    Attributes
    ----------
    QUERY_AND_TARGET
        The alignment must cover the given fraction of both sequences.
    TARGET
        The alignment must cover the given fraction of the target sequence.
    QUERY
        The alignment must cover the given fraction of the query sequence.
    TARGET_LENGTH_IN_QUERY
        The target sequence must have the given fraction of the query length.
    QUERY_LENGTH_IN_TARGET
        The query sequence must have the given fraction of the target length.
    SHORTER_IN_LONGER
        The shorter sequence must have the given fraction of the longer length.
    """

    QUERY_AND_TARGET = 0
    TARGET = 1
    QUERY = 2
    TARGET_LENGTH_IN_QUERY = 3
    QUERY_LENGTH_IN_TARGET = 4
    SHORTER_IN_LONGER = 5


class ClusterMode(IntEnum):
    """
    Values accepted by the ``cluster_mode`` option.

    Attributes
    ----------
    SET_COVER
        Greedy set cover, which minimizes the number of clusters.
    CONNECTED_COMPONENT
        Transitive clustering, as performed by *BLASTclust*.
    GREEDY_BY_LENGTH
        Greedy clustering around the longest sequences, as performed by *CD-HIT*.
    GREEDY_BY_LENGTH_LOW_MEMORY
        Like :attr:`GREEDY_BY_LENGTH`, but with a lower memory footprint.
    """

    SET_COVER = 0
    CONNECTED_COMPONENT = 1
    GREEDY_BY_LENGTH = 2
    GREEDY_BY_LENGTH_LOW_MEMORY = 3


class SequenceIdentityMode(IntEnum):
    """
    Reference lengths accepted by the ``seq_id_mode`` option.

    The sequence identity is the number of identical residues divided by the
    selected reference length.

    Attributes
    ----------
    ALIGNMENT_LENGTH
        The number of aligned columns.
    SHORTER_SEQUENCE
        The length of the shorter of the two sequences.
    LONGER_SEQUENCE
        The length of the longer of the two sequences.
    """

    ALIGNMENT_LENGTH = 0
    SHORTER_SEQUENCE = 1
    LONGER_SEQUENCE = 2


class StructureAlignmentType(IntEnum):
    """
    Values accepted by the ``alignment_type`` option of :class:`FoldseekApp`.

    Attributes
    ----------
    I3D
        Local alignment of the *3Di* sequences alone.
    TM_ALIGN
        Global *TMalign* alignment of the coordinates.
    I3D_AND_AMINO_ACID
        Local alignment of the *3Di* and amino acid sequences combined.
    """

    I3D = 0
    TM_ALIGN = 1
    I3D_AND_AMINO_ACID = 2


class UnpackNameMode(IntEnum):
    """
    File naming schemes accepted by the ``unpack_name_mode`` option.

    Attributes
    ----------
    DATABASE_KEY
        Name each file after the numeric database key of its entry.
    ACCESSION
        Name each file after the identifier of its entry, which requires the
        database to have a lookup file.
    """

    DATABASE_KEY = 0
    ACCESSION = 1


_COMMON_OPTIONS = ["threads", "compressed", "v"]
_DATABASES_OPTIONS = [*_COMMON_OPTIONS, "force_reuse", "remove_tmp_files"]
_CREATE_DB_OPTIONS = [
    *_COMMON_OPTIONS,
    "shuffle",
    "id_offset",
    "write_lookup",
    "mask",
    "mask_prob",
    "mask_lower_case",
    "mask_n_repeat",
    "mask_bfactor_threshold",
    "input_format",
    "file_include",
    "file_exclude",
    "prostt5_model",
    "chain_name_mode",
    "write_mapping",
    "coord_store_mode",
    "db_extraction_mode",
    "distance_threshold",
    "gpu",
]
_CREATE_SUB_DB_OPTIONS = ["v", "id_mode"]
_CREATE_INDEX_OPTIONS = [
    *_COMMON_OPTIONS,
    "seed_sub_mat",
    "k",
    "alph_size",
    "comp_bias_corr",
    "comp_bias_corr_scale",
    "max_seqs",
    "mask",
    "mask_prob",
    "mask_lower_case",
    "spaced_kmer_mode",
    "spaced_kmer_pattern",
    "s",
    "k_score",
    "split",
    "split_memory_limit",
    "check_compatible",
    "search_type",
    "min_length",
    "max_length",
    "max_gaps",
    "contig_start_mode",
    "contig_end_mode",
    "orf_start_mode",
    "forward_frames",
    "reverse_frames",
    "translation_table",
    "translate",
    "use_all_table_starts",
    "add_orf_stop",
    "id_offset",
    "sequence_overlap",
    "sequence_split_mode",
    "headers_split_mode",
    "translation_mode",
    "max_seq_len",
    "remove_tmp_files",
    "create_lookup",
    "strand",
]
_CONVERT_TO_FASTA_OPTIONS = ["v", "use_header_file"]
_CONVERT_ALIGNMENTS_OPTIONS = [
    *_COMMON_OPTIONS,
    "gap_open",
    "gap_extend",
    "format_output",
    "translation_table",
    "search_type",
    "exact_tmscore",
    "sub_mat",
    "db_load_mode",
]
_CONVERT_MSA_OPTIONS = ["compressed", "v", "identifier_field"]
_UNPACK_DB_OPTIONS = ["threads", "v", "unpack_name_mode", "unpack_suffix"]
_CREATE_TSV_OPTIONS = [
    *_COMMON_OPTIONS,
    "first_seq_as_repr",
    "target_column",
    "full_header",
    "idx_seq_src",
]
# The options of the gapped alignment stage, shared by all commands that align
_ALIGN_OPTIONS = [
    *_COMMON_OPTIONS,
    "comp_bias_corr",
    "comp_bias_corr_scale",
    "add_self_matches",
    "a",
    "alignment_mode",
    "wrapped_scoring",
    "e",
    "min_seq_id",
    "min_aln_len",
    "seq_id_mode",
    "alt_ali",
    "c",
    "cov_mode",
    "max_rejected",
    "max_accept",
    "score_bias",
    "realign",
    "realign_score_bias",
    "realign_max_seqs",
    "corr_score_weight",
    "gap_open",
    "gap_extend",
    "zdrop",
    "pca",
    "pcb",
    "sub_mat",
    "max_seq_len",
    "db_load_mode",
]
# The *k-mer* stage, shared by the search and clustering workflows
_KMER_OPTIONS = [
    "alph_size",
    "spaced_kmer_mode",
    "spaced_kmer_pattern",
    "mask",
    "mask_prob",
    "mask_lower_case",
    "k",
    "split_memory_limit",
    "rescore_mode",
    "remove_tmp_files",
    "force_reuse",
    "mpi_runner",
    "filter_hits",
    "sort_results",
]
# The clustering criteria, shared by both clustering workflows
_CLUSTERING_OPTIONS = [
    "cluster_mode",
    "max_iterations",
    "similarity_type",
    "weights",
    "cluster_weight_threshold",
    "kmer_per_seq",
    "kmer_per_seq_scale",
    "adjust_kmer_len",
    "hash_shift",
    "include_only_extendable",
    "ignore_multi_kmer",
]
_LINCLUST_OPTIONS = [*_ALIGN_OPTIONS, *_KMER_OPTIONS, *_CLUSTERING_OPTIONS]
_CLUSTER_OPTIONS = [
    *_LINCLUST_OPTIONS,
    "seed_sub_mat",
    "s",
    "target_search_mode",
    "k_score",
    "max_seqs",
    "split",
    "split_mode",
    "diag_score",
    "exact_kmer_matching",
    "min_ungapped_score",
    "single_step_clustering",
    "cluster_steps",
    "cluster_reassign",
    "taxon_list",
]
_SEARCH_OPTIONS = [
    *_ALIGN_OPTIONS,
    *_KMER_OPTIONS,
    "seed_sub_mat",
    "s",
    "target_search_mode",
    "k_score",
    "max_seqs",
    "split",
    "split_mode",
    "diag_score",
    "exact_kmer_matching",
    "min_ungapped_score",
    "disk_space_limit",
    "exhaustive_search_filter",
    "mask_profile",
    "e_profile",
    "wg",
    "filter_msa",
    "filter_min_enable",
    "max_seq_id",
    "qid",
    "qsc",
    "cov",
    "diff",
    "pseudo_cnt_mode",
    "num_iterations",
    "exhaustive_search",
    "lca_search",
    "taxon_list",
    "prefilter_mode",
    "allow_deletion",
    "min_length",
    "max_length",
    "max_gaps",
    "contig_start_mode",
    "contig_end_mode",
    "orf_start_mode",
    "forward_frames",
    "reverse_frames",
    "translation_table",
    "translate",
    "use_all_table_starts",
    "id_offset",
    "sequence_overlap",
    "sequence_split_mode",
    "headers_split_mode",
    "search_type",
    "start_sens",
    "sens_steps",
    "translation_mode",
    "gpu",
    "gpu_server",
    "create_lookup",
    "chain_alignments",
    "merge_query",
    "strand",
]

# The *Foldseek* counterparts of the option groups above
_STRUCTURE_ALIGN_OPTIONS = [
    *_COMMON_OPTIONS,
    "comp_bias_corr",
    "comp_bias_corr_scale",
    "sort_by_structure_bits",
    "a",
    "alignment_mode",
    "e",
    "min_seq_id",
    "min_aln_len",
    "seq_id_mode",
    "alt_ali",
    "c",
    "cov_mode",
    "max_rejected",
    "max_accept",
    "gap_open",
    "gap_extend",
    "tmscore_threshold",
    "lddt_threshold",
    "alignment_type",
    "exact_tmscore",
    "sub_mat",
    "max_seq_len",
    "db_load_mode",
]
_STRUCTURE_KMER_OPTIONS = [
    "seed_sub_mat",
    "s",
    "k",
    "target_search_mode",
    "k_score",
    "max_seqs",
    "split",
    "split_mode",
    "split_memory_limit",
    "diag_score",
    "exact_kmer_matching",
    "mask",
    "mask_prob",
    "mask_lower_case",
    "min_ungapped_score",
    "spaced_kmer_mode",
    "spaced_kmer_pattern",
    "tmalign_hit_order",
    "tmalign_fast",
    "remove_tmp_files",
    "mpi_runner",
    "force_reuse",
    "taxon_list",
]
_STRUCTURE_SEARCH_OPTIONS = [
    *_STRUCTURE_ALIGN_OPTIONS,
    *_STRUCTURE_KMER_OPTIONS,
    "exhaustive_search",
    "num_iterations",
    "prefilter_mode",
    "cluster_search",
]
_STRUCTURE_CLUSTER_OPTIONS = [
    *_STRUCTURE_ALIGN_OPTIONS,
    *_STRUCTURE_KMER_OPTIONS,
    "cluster_mode",
    "max_iterations",
    "similarity_type",
    "single_step_clustering",
    "cluster_steps",
    "cluster_reassign",
    "weights",
    "cluster_weight_threshold",
    "kmer_per_seq",
    "kmer_per_seq_scale",
    "adjust_kmer_len",
    "hash_shift",
    "include_only_extendable",
    "ignore_multi_kmer",
    "rescore_mode",
    "zdrop",
    "filter_hits",
    "sort_results",
]
_MULTIMER_SEARCH_OPTIONS = [
    *_STRUCTURE_SEARCH_OPTIONS,
    "expand_multimer_evalue",
    "min_assigned_chains_ratio",
    "zdrop",
]

# The MSA diversity filter, applied by every command that summarizes a result set
_MSA_FILTER_OPTIONS = [
    "comp_bias_corr",
    "comp_bias_corr_scale",
    "cov",
    "diff",
    "filter_min_enable",
    "filter_msa",
    "gap_extend",
    "gap_open",
    "max_seq_id",
    "qid",
    "qsc",
    "sub_mat",
]
# The pseudo count admixture, applied by every command that creates a profile
_PSEUDO_COUNT_OPTIONS = ["pca", "pcb", "pseudo_cnt_mode", "wg"]
_MULTIMER_REPORT_OPTIONS = ["threads", "v"]
_RESULT_TO_MSA_OPTIONS = [
    *_COMMON_OPTIONS,
    *_MSA_FILTER_OPTIONS,
    "allow_deletion",
    "db_load_mode",
    "summary_prefix",
    "skip_query",
]
_RESULT_TO_PROFILE_OPTIONS = [
    *_COMMON_OPTIONS,
    *_MSA_FILTER_OPTIONS,
    *_PSEUDO_COUNT_OPTIONS,
    "e",
    "e_profile",
    "mask_profile",
    "allow_deletion",
    "db_load_mode",
]
_MSA_TO_PROFILE_OPTIONS = [
    *_COMMON_OPTIONS,
    *_MSA_FILTER_OPTIONS,
    *_PSEUDO_COUNT_OPTIONS,
    "match_ratio",
    "msa_type",
    "skip_query",
]
_PROFILE_TO_SEQUENCES_OPTIONS = [*_COMMON_OPTIONS, "sub_mat", "max_seq_len"]
_PROFILE_TO_PSSM_OPTIONS = [
    *_PROFILE_TO_SEQUENCES_OPTIONS,
    "comp_bias_corr",
    "comp_bias_corr_scale",
]


class MMseqsLikeApp(LocalApp):
    """
    Shared command implementation for MMseqs2-compatible applications.

    Parameters
    ----------
    path : path-like
        Path to the application executable.
    """

    _tmp_prefix: ClassVar[str]

    def __init__(self, path: PathLike[str] | str) -> None:
        super().__init__(path)
        self._tmp_dir = Path(gettempdir())

    @property
    def tmp_dir(self) -> Path:
        """
        The temporary directory required for some *MMseqs2* and *Foldseek*
        commands.

        Returns
        -------
        tmp_dir : Path
            The parent directory for temporary command workspaces.
        """
        return self._tmp_dir

    @tmp_dir.setter
    def tmp_dir(self, tmp_dir: PathLike[str] | str) -> None:
        """
        Set the temporary directory required for some *MMseqs2* and *Foldseek*
        commands.

        Parameters
        ----------
        tmp_dir : path-like
            The parent directory for temporary command workspaces.
        """
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(exist_ok=True, parents=True)
        self._tmp_dir = tmp_dir

    def _format_value(self, value: Any) -> str:
        match value:
            case Database():
                if not value.is_compatible_with(self):
                    raise ValueError(
                        f"Database is not compatible with {type(self).__name__}"
                    )
                return str(value.name)
            case tuple() | list():
                return ",".join(self._format_value(element) for element in value)
            case _:
                return super()._format_value(value)

    @command(subcommand="databases", allowed_options=_DATABASES_OPTIONS)
    def databases(
        self,
        name: str,
        **kwargs: Any,
    ) -> CommandSetup[SequenceDatabase[Any]]:
        """
        Download and prepare a named database using ``databases``.

        Parameters
        ----------
        name : str
            The database name understood by the application.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of SequenceDatabase
            A handle resolving to the downloaded sequence database.

        Examples
        --------
        >>> database = MMseqsApp().databases("UniRef50").result()  # doctest: +SKIP
        """
        database = SequenceDatabase(self)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                CLIArgument(name),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="createdb", allowed_options=_CREATE_DB_OPTIONS)
    def create_db(
        self,
        input_paths: (
            Iterable[PathLike[str] | str | IO[str]] | PathLike[str] | str | IO[str]
        ),
        db_type: DatabaseType | None = None,
        **kwargs: Any,
    ) -> CommandSetup[SequenceDatabase[Any]]:
        """
        Create a sequence or structure database using ``createdb``.

        Parameters
        ----------
        input_paths : path-like, text file or iterable thereof
            FASTA/FASTQ inputs for MMseqs2, or structural inputs for Foldseek.
            A text file must expose its file-system path through its ``name``
            attribute, and pending writes must be flushed before calling this
            method.
        db_type : DatabaseType, optional
            The type of the input sequences. By default, *MMseqs2* detects the
            type automatically. This option is not supported by *Foldseek*.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of SequenceDatabase
            A handle resolving to the created sequence database.

        Examples
        --------
        >>> database = MMseqsApp().create_db(path_to_sequences / "prot.fasta").result()
        """
        if isinstance(input_paths, (str, PathLike, IOBase)):
            inputs = [input_paths]
        else:
            inputs = list(input_paths)
        if len(inputs) == 0:
            raise ValueError("At least one input is required")
        database = SequenceDatabase(self)
        parameters: list[CLIArgument | CLIOption] = [
            *(
                CLIArgument(
                    Path(input) if isinstance(input, (str, PathLike)) else input
                )
                for input in inputs
            ),
            CLIArgument(database),
        ]
        if db_type is not None:
            parameters.append(CLIOption("dbtype", db_type))
        return CommandSetup(
            parameters=parameters,
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="createsubdb", allowed_options=_CREATE_SUB_DB_OPTIONS)
    def create_sub_db(
        self,
        subset: Database[Any] | PathLike[str] | str,
        sequence_db: SequenceDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[SequenceDatabase[Any]]:
        """
        Create a sequence database containing a selected subset using
        ``createsubdb``.

        Parameters
        ----------
        subset : Database or path-like
            A database or file defining the retained database keys.
        sequence_db : SequenceDatabase
            The source sequence database.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of SequenceDatabase
            A handle resolving to the created subset database.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "prot.fasta").result()
        >>> # Trivial case: Select all entries from the original database
        >>> subset = app.create_sub_db(database, database).result()
        """
        database = SequenceDatabase(self)
        return CommandSetup(
            parameters=[
                CLIArgument(subset),
                CLIArgument(sequence_db),
                CLIArgument(database),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="createindex", allowed_options=_CREATE_INDEX_OPTIONS)
    def create_index(
        self,
        sequence_db: SequenceDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[SequenceDatabase[Any]]:
        """
        Precompute the *k-mer* index of a sequence database using
        ``createindex``.

        Parameters
        ----------
        sequence_db : SequenceDatabase
            The database to create the index for.
            The index is added in place to this database.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of SequenceDatabase
            A handle resolving to the indexed database.
            As the index is added in place, this is the same database object
            as `sequence_db`.

        Notes
        -----
        Without a precomputed index, each :meth:`search()` call computes it anew.
        Hence, creating it once accelerates repeated searches against the same
        database.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> indexed_database = app.create_index(database, threads=1).result()
        >>> indexed_database is database
        True
        """
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[CLIArgument(sequence_db), CLIArgument(work_dir)],
            evaluate=lambda _out, _err: sequence_db,
        )

    @overload
    def convert_to_fasta(
        self,
        sequence_db: SequenceDatabase[Any],
        fasta_path: None = None,
        **kwargs: Any,
    ) -> LocalProcessFuture[IO[bytes]]: ...

    @overload
    def convert_to_fasta(
        self,
        sequence_db: SequenceDatabase[Any],
        fasta_path: PathLike[str] | str,
        **kwargs: Any,
    ) -> LocalProcessFuture[Path]: ...

    @command(subcommand="convert2fasta", allowed_options=_CONVERT_TO_FASTA_OPTIONS)
    def convert_to_fasta(
        self,
        sequence_db: SequenceDatabase[Any],
        fasta_path: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[Path | IO[bytes]]:
        """
        Export a sequence database in FASTA format using ``convert2fasta``.

        Parameters
        ----------
        sequence_db : SequenceDatabase
            The database to export.
        fasta_path : path-like, optional
            The output FASTA path. If omitted, the output is written to a
            temporary binary file.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path or binary file
            A handle resolving to the output path, or to the temporary file if
            `fasta_path` is omitted.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "prot.fasta").result()
        >>> fasta_file = app.convert_to_fasta(database).result()
        >>> fasta_file.read().startswith(b">")
        True
        >>> fasta_file.close()
        """
        output: Path | IO[bytes]
        if fasta_path is None:
            output = NamedTemporaryFile("w+b")
        else:
            output = Path(fasta_path)
        return CommandSetup(
            parameters=[CLIArgument(sequence_db), CLIArgument(output)],
            evaluate=lambda _out, _err: output,
        )

    @overload
    def convert_alignments(
        self,
        alignment_db: AlignmentDatabase[Any],
        output_path: None = None,
        format_mode: AlignmentFormatMode | None = None,
        **kwargs: Any,
    ) -> LocalProcessFuture[IO[bytes]]: ...

    @overload
    def convert_alignments(
        self,
        alignment_db: AlignmentDatabase[Any],
        output_path: PathLike[str] | str,
        format_mode: AlignmentFormatMode | None = None,
        **kwargs: Any,
    ) -> LocalProcessFuture[Path]: ...

    @command(subcommand="convertalis", allowed_options=_CONVERT_ALIGNMENTS_OPTIONS)
    def convert_alignments(
        self,
        alignment_db: AlignmentDatabase[Any],
        output_path: PathLike[str] | str | None = None,
        format_mode: AlignmentFormatMode | None = None,
        **kwargs: Any,
    ) -> CommandSetup[Path | IO[bytes]]:
        """
        Export pairwise alignments in a human-readable format using ``convertalis``.

        Parameters
        ----------
        alignment_db : AlignmentDatabase
            The alignment database to export.
        output_path : path-like, optional
            The output path. If omitted, the output is written to a temporary
            binary file.
        format_mode : AlignmentFormatMode, optional
            The output format. By default, the application writes a BLAST-like
            table.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path or binary file
            A handle resolving to the output path, or to the temporary file if
            `output_path` is omitted.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "prot.fasta").result()
        >>> alignments = app.search(database, database, threads=1).result()
        >>> alignment_file = app.convert_alignments(alignments).result()
        >>> len(alignment_file.readline()) > 0
        True
        >>> alignment_file.close()
        """
        output: Path | IO[bytes]
        if output_path is None:
            output = NamedTemporaryFile("w+b")
        else:
            output = Path(output_path)
        parameters: list[CLIArgument | CLIOption] = [
            CLIArgument(alignment_db.query_db),
            CLIArgument(alignment_db.target_db),
            CLIArgument(alignment_db),
            CLIArgument(output),
        ]
        if format_mode is not None:
            parameters.append(CLIOption("format_mode", format_mode))
        return CommandSetup(
            parameters=parameters,
            evaluate=lambda _out, _err: output,
        )

    @command(subcommand="result2msa", allowed_options=_RESULT_TO_MSA_OPTIONS)
    def result_to_msa(
        self,
        alignment_db: AlignmentDatabase[Any],
        msa_format_mode: MSAFormatMode | None = None,
        **kwargs: Any,
    ) -> CommandSetup[MSADatabase[Any]]:
        """
        Convert pairwise alignment results into multiple alignments using
        ``result2msa``.

        Parameters
        ----------
        alignment_db : AlignmentDatabase
            The pairwise alignment results.
        msa_format_mode : MSAFormatMode, optional
            The multiple alignment format.
            The flat-file :attr:`MSAFormatMode.STOCKHOLM` mode is not supported because
            this method returns an :class:`MSADatabase`.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of MSADatabase
            A handle resolving to the created MSA database.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "prot.fasta").result()
        >>> alignments = app.search(database, database, threads=1).result()
        >>> msa_database = app.result_to_msa(alignments, threads=1).result()
        """
        if msa_format_mode == MSAFormatMode.STOCKHOLM:
            raise ValueError(
                "The STOCKHOLM format produces a flat file instead of an MSADatabase"
            )
        database = MSADatabase(self)
        parameters: list[CLIArgument | CLIOption] = [
            CLIArgument(alignment_db.query_db),
            CLIArgument(alignment_db.target_db),
            CLIArgument(alignment_db),
            CLIArgument(database),
        ]
        if msa_format_mode is not None:
            parameters.append(CLIOption("msa_format_mode", msa_format_mode))
        return CommandSetup(
            parameters=parameters,
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="lndb", allowed_options=["v"])
    def link_db(
        self,
        source: PathLike[str] | str,
        destination: PathLike[str] | str,
        **kwargs: Any,
    ) -> CommandSetup[Path]:
        """
        Create a symbolic database link using ``lndb``.

        This is an advanced feature.
        For more details consult the
        `MMseqs2 database format <https://github.com/soedinglab/MMseqs2/wiki#mmseqs2-database-format>`_

        Parameters
        ----------
        source : path-like
            The source database prefix.
        destination : path-like
            The destination database prefix.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path
            A handle resolving to the destination prefix.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "prot.fasta").result()
        >>> with tempfile.NamedTemporaryFile() as link_file:
        ...     link_path = Path(link_file.name)
        >>> _ = app.link_db(database.name, link_path).result()
        >>> link_path.is_symlink()
        True
        >>> link_path.unlink()
        """
        source = Path(source)
        destination = Path(destination)
        return CommandSetup(
            parameters=[CLIArgument(source), CLIArgument(destination)],
            evaluate=lambda _out, _err: destination,
        )

    @command(subcommand="unpackdb", allowed_options=_UNPACK_DB_OPTIONS)
    def unpack_db(
        self,
        database: Database[Any],
        directory: PathLike[str] | str,
        **kwargs: Any,
    ) -> CommandSetup[Path]:
        """
        Write each entry of a database into a separate file using ``unpackdb``.

        This is an advanced feature.
        For more details consult the
        `MMseqs2 database format <https://github.com/soedinglab/MMseqs2/wiki#mmseqs2-database-format>`_

        Parameters
        ----------
        database : Database
            The database to unpack.
            Any type of database is accepted, as the entry content is copied
            verbatim.
        directory : path-like
            The directory the files are written to.
            It is created, if it does not exist.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path
            A handle resolving to the directory containing the written files.

        Notes
        -----
        A database stores all of its entries concatenated in a single data file,
        addressed by byte offsets in an accompanying index.
        This command reverses that packing: it writes the raw content of each
        entry into its own file, without interpreting it.
        Hence it is the general way to read entries of a database that no
        dedicated conversion command covers.

        By default each file is named after the identifier of its entry, taken
        from the lookup file of the database.
        Only databases created from named input have such a file, so for a
        *sub-database* such as the header database, and for result databases such
        as an :class:`AlignmentDatabase`, the numeric database key is used
        instead.
        The ``unpack_name_mode`` option selects the scheme explicitly (see
        :class:`UnpackNameMode`) and ``unpack_suffix`` appends a file extension.

        The written content is the entry data with its terminating null byte
        removed.
        A trailing newline that is part of the entry is kept, so for a sequence
        database each file contains the sequence followed by a line break.
        What an entry comprises depends on the database: a single sequence for a
        :class:`SequenceDatabase`, or all hits of one query for an
        :class:`AlignmentDatabase`.
        Note that the latter are written in the internal representation, which
        refers to targets by their database key, hence
        :meth:`convert_alignments()` is usually the better choice for reading
        alignments.

        Examples
        --------
        >>> import tempfile
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "prot.fasta").result()
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     unpacked = app.unpack_db(database, directory, threads=1).result()
        ...     print(sorted(path.name for path in unpacked.iterdir()))
        ['protein']
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        return CommandSetup(
            parameters=[CLIArgument(database), CLIArgument(directory)],
            evaluate=lambda _out, _err: directory,
        )

    @overload
    def create_tsv(
        self,
        result_db: AlignmentDatabase[Any] | ClusterDatabase[Any],
        output_path: None = None,
        **kwargs: Any,
    ) -> LocalProcessFuture[IO[bytes]]: ...

    @overload
    def create_tsv(
        self,
        result_db: AlignmentDatabase[Any] | ClusterDatabase[Any],
        output_path: PathLike[str] | str,
        **kwargs: Any,
    ) -> LocalProcessFuture[Path]: ...

    @command(subcommand="createtsv", allowed_options=_CREATE_TSV_OPTIONS)
    def create_tsv(
        self,
        result_db: AlignmentDatabase[Any] | ClusterDatabase[Any],
        output_path: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[Path | IO[bytes]]:
        """
        Export a result database as a tab-separated file using ``createtsv``.

        In contrast to :meth:`convert_alignments()` the identifiers are taken
        from the input databases, which makes this the way to read the result of
        :meth:`MMseqsApp.cluster()`.

        Parameters
        ----------
        result_db : AlignmentDatabase or ClusterDatabase
            The result database to export.
        output_path : path-like, optional
            The output path. If omitted, the output is written to a temporary
            binary file.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path or binary file
            A handle resolving to the output path, or to the temporary file if
            `output_path` is omitted.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> clusters = app.cluster(database, threads=1).result()
        >>> tsv_file = app.create_tsv(clusters, threads=1).result()
        >>> print(tsv_file.readline().decode().split()[0])
        sp|Q99ZW2.1|CAS9_STRP1
        >>> tsv_file.close()
        """
        if isinstance(result_db, ClusterDatabase):
            query_db: Database[Any] = result_db.sequence_db
            target_db: Database[Any] = result_db.sequence_db
        else:
            query_db = result_db.query_db
            target_db = result_db.target_db
        output: Path | IO[bytes]
        if output_path is None:
            output = NamedTemporaryFile("w+b")
        else:
            output = Path(output_path)
        return CommandSetup(
            parameters=[
                CLIArgument(query_db),
                CLIArgument(target_db),
                CLIArgument(result_db),
                CLIArgument(output),
            ],
            evaluate=lambda _out, _err: output,
        )

    def _create_work_dir(self) -> TemporaryDirectory[str]:
        """
        Create an isolated command workspace below the temporary directory.

        Returns
        -------
        TemporaryDirectory
            The created temporary directory.
        """
        tmp_dir = self.tmp_dir
        tmp_dir.mkdir(exist_ok=True, parents=True)
        return TemporaryDirectory(prefix=self._tmp_prefix, dir=tmp_dir)


class MMseqsApp(MMseqsLikeApp):
    """
    A reusable handle to the *MMseqs2* command line program.

    Parameters
    ----------
    path : path-like, optional
        Path to the ``mmseqs`` executable.

    Attributes
    ----------
    tmp_dir : Path
        The parent directory for temporary command workspaces.

    Examples
    --------
    >>> future = MMseqsApp().create_db(path_to_sequences / "prot.fasta")
    >>> isinstance(future.result(), SequenceDatabase)
    True
    """

    _tmp_prefix = "mmseqs_"

    def __init__(self, path: PathLike[str] | str = "mmseqs") -> None:
        super().__init__(path)

    @command(subcommand="search", allowed_options=_SEARCH_OPTIONS)
    def search(
        self,
        query_db: SequenceDatabase[Any] | ProfileDatabase[Any],
        target_db: SequenceDatabase[Any],
        matrix: SubstitutionMatrix[Any, Any] | None = None,
        gap_penalty: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Search a target database for matches to every query using ``search``.

        Parameters
        ----------
        query_db : SequenceDatabase or ProfileDatabase
            The query database.
            If a :class:`ProfileDatabase` is given, a more sensitive
            profile-to-sequence search is performed.
        target_db : SequenceDatabase
            The target database.
        matrix : SubstitutionMatrix, optional
            A custom symmetric substitution matrix.
            By default, the standard *MMseqs2* matrix is used.
        gap_penalty : int or tuple of (int, int), optional
            The negative gap penalty.
            A single value applies to both opening and extension.
            A tuple gives the respective affine penalties.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            search results.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> matrix = SubstitutionMatrix.std_protein_matrix()
        >>> alignment_db = app.search(
        ...     database, database, matrix=matrix,
        ...     gap_penalty=(-11, -1), threads=1,
        ... ).result()
        """
        options = _resolve_scoring(matrix, gap_penalty)
        database = AlignmentDatabase(self, query_db, target_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                *options,
                CLIArgument(query_db),
                CLIArgument(target_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="convertmsa", allowed_options=_CONVERT_MSA_OPTIONS)
    def convert_msa(
        self,
        msa_path: PathLike[str] | str | IO[str],
        **kwargs: Any,
    ) -> CommandSetup[MSADatabase[Any]]:
        """
        Create an MSA database from a *Stockholm* file using ``convertmsa``.

        Parameters
        ----------
        msa_path : path-like or text file
            The MSA file to convert.
            Must be in *Stockholm* format, optionally gzip-compressed.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of MSADatabase
            A handle resolving to the created MSA database.

        Examples
        --------
        >>> import tempfile
        >>> app = MMseqsApp()
        >>> with tempfile.NamedTemporaryFile("w", suffix=".sto") as msa_file:
        ...     print("# STOCKHOLM 1.0", file=msa_file)
        ...     print("#=GF AC query", file=msa_file)
        ...     print("seq0 NLYIQWLKDGGPSSGRPPPS", file=msa_file)
        ...     print("seq1 NLYIQWLKD-GPSSGRPPPA", file=msa_file)
        ...     print("//", file=msa_file)
        ...     msa_file.flush()
        ...     msa_db = app.convert_msa(msa_file).result()
        """
        database = MSADatabase(self)
        return CommandSetup(
            parameters=[
                CLIArgument(
                    Path(msa_path)
                    if isinstance(msa_path, (str, PathLike))
                    else msa_path
                ),
                CLIArgument(database),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="msa2profile", allowed_options=_MSA_TO_PROFILE_OPTIONS)
    def msa_to_profile(
        self,
        msa_db: MSADatabase[Any],
        match_mode: ProfileMatchMode | None = None,
        **kwargs: Any,
    ) -> CommandSetup[ProfileDatabase[Any]]:
        """
        Convert an MSA database into a profile database using ``msa2profile``.

        Parameters
        ----------
        msa_db : MSADatabase
            The MSA database to create the profiles from.
        match_mode : ProfileMatchMode, optional
            Defines which MSA columns become profile columns.
            By default, the columns are defined by the first sequence of each
            MSA, i.e. the profile is kept in the coordinate frame of the query sequence.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ProfileDatabase
            A handle resolving to the created profile database.

        Examples
        --------
        >>> a3m_file = FastaFile.read(path_to_sequences / "1a00_A_uniref90.a3m")
        >>> msa = get_a3m_alignments(a3m_file)
        >>> app = MMseqsApp()
        >>> msa_db = create_database_from_msa(app, {"query": msa})
        >>> profile_db = app.msa_to_profile(msa_db, threads=1).result()
        """
        database = ProfileDatabase(self)
        parameters: list[CLIArgument | CLIOption] = [
            CLIArgument(msa_db),
            CLIArgument(database),
        ]
        if match_mode is not None:
            parameters.append(CLIOption("match_mode", match_mode))
        return CommandSetup(
            parameters=parameters,
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="profile2repseq", allowed_options=_PROFILE_TO_SEQUENCES_OPTIONS)
    def profile_to_sequences(
        self,
        profile_db: ProfileDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[SequenceDatabase[Any]]:
        """
        Extract the representative sequence of each profile using
        ``profile2repseq``.

        Parameters
        ----------
        profile_db : ProfileDatabase
            The profile database to extract the sequences from.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of SequenceDatabase
            A handle resolving to the sequence database containing the
            representative sequences.

        Notes
        -----
        For a profile created via :meth:`msa_to_profile()` the representative
        sequence is the first sequence of the MSA, i.e. the query sequence for
        MSAs created via :meth:`result_to_msa()`.

        Examples
        --------
        >>> a3m_file = FastaFile.read(path_to_sequences / "1a00_A_uniref90.a3m")
        >>> msa = get_a3m_alignments(a3m_file)
        >>> app = MMseqsApp()
        >>> msa_db = create_database_from_msa(app, {"query": msa})
        >>> profile_db = app.msa_to_profile(msa_db, threads=1).result()
        >>> sequence_db = app.profile_to_sequences(profile_db, threads=1).result()
        >>> # The representative sequence of the profile is the query of the MSA
        >>> sequences = get_sequences_from_database(app, sequence_db)
        >>> sequences["query"] == str(msa[0].sequences[0])
        True
        """
        database = SequenceDatabase(self)
        return CommandSetup(
            parameters=[CLIArgument(profile_db), CLIArgument(database)],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="align", allowed_options=_ALIGN_OPTIONS)
    def align(
        self,
        alignment_db: AlignmentDatabase[Any],
        matrix: SubstitutionMatrix[Any, Any] | None = None,
        gap_penalty: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Recompute the gapped alignments of a result database using ``align``.

        This allows realigning the hits found by :meth:`search()` with different
        alignment parameters, without repeating the expensive prefilter stage.

        Parameters
        ----------
        alignment_db : AlignmentDatabase
            The result database whose hits are aligned.
        matrix : SubstitutionMatrix, optional
            A custom symmetric substitution matrix.
            By default, the standard *MMseqs2* matrix is used.
        gap_penalty : int or tuple of (int, int), optional
            The negative gap penalty.
            A single value applies to both opening and extension.
            A tuple gives the respective affine penalties.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            recomputed alignments.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> alignment_db = app.search(database, database, threads=1).result()
        >>> realigned_db = app.align(alignment_db, gap_penalty=(-15, -2), threads=1).result()
        """
        database = AlignmentDatabase(
            self, alignment_db.query_db, alignment_db.target_db
        )
        return CommandSetup(
            parameters=[
                *_resolve_scoring(matrix, gap_penalty),
                CLIArgument(alignment_db.query_db),
                CLIArgument(alignment_db.target_db),
                CLIArgument(alignment_db),
                CLIArgument(database),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="rbh", allowed_options=_SEARCH_OPTIONS)
    def search_reciprocal_best_hits(
        self,
        query_db: SequenceDatabase[Any],
        target_db: SequenceDatabase[Any],
        matrix: SubstitutionMatrix[Any, Any] | None = None,
        gap_penalty: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Find reciprocal best hits between two databases using ``rbh``.

        Only hits that are the best hit in both search directions are reported.

        Parameters
        ----------
        query_db, target_db : SequenceDatabase
            The query and target databases.
        matrix : SubstitutionMatrix, optional
            A custom symmetric substitution matrix.By default, the standard
            *MMseqs2* matrix is used.
        gap_penalty : int or tuple of (int, int), optional
            The negative gap penalty.
            A single value applies to both opening and extension.
            A tuple gives the respective affine penalties.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            reciprocal best hits.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> hits = app.search_reciprocal_best_hits(database, database, threads=1).result()
        """
        database = AlignmentDatabase(self, query_db, target_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                *_resolve_scoring(matrix, gap_penalty),
                CLIArgument(query_db),
                CLIArgument(target_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="cluster", allowed_options=_CLUSTER_OPTIONS)
    def cluster(
        self,
        sequence_db: SequenceDatabase[Any],
        matrix: SubstitutionMatrix[Any, Any] | None = None,
        gap_penalty: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[ClusterDatabase[Any]]:
        """
        Cluster a sequence database using the cascaded ``cluster`` workflow.

        Parameters
        ----------
        sequence_db : SequenceDatabase
            The database to cluster.
        matrix : SubstitutionMatrix, optional
            A custom symmetric substitution matrix.
            By default, the standard *MMseqs2* matrix is used.
        gap_penalty : int or tuple of (int, int), optional
            The negative gap penalty.
            A single value applies to both opening and extension.
            A tuple gives the respective affine penalties.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ClusterDatabase
            A handle resolving to the created cluster database.

        See Also
        --------
        cluster_linear
            The faster but less sensitive alternative.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> clusters = app.cluster(database, min_seq_id=0.9, threads=1).result()
        >>> isinstance(clusters, ClusterDatabase)
        True
        """
        database = ClusterDatabase(self, sequence_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                *_resolve_scoring(matrix, gap_penalty),
                CLIArgument(sequence_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="linclust", allowed_options=_LINCLUST_OPTIONS)
    def cluster_linear(
        self,
        sequence_db: SequenceDatabase[Any],
        matrix: SubstitutionMatrix[Any, Any] | None = None,
        gap_penalty: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> CommandSetup[ClusterDatabase[Any]]:
        """
        Cluster a sequence database using the ``linclust`` workflow.

        The run time is linear in the database size, at the expense of
        sensitivity compared to :meth:`cluster()`.

        Parameters
        ----------
        sequence_db : SequenceDatabase
            The database to cluster.
        matrix : SubstitutionMatrix, optional
            A custom symmetric substitution matrix.
            By default, the standard *MMseqs2* matrix is used.
        gap_penalty : int or tuple of (int, int), optional
            The negative gap penalty.
            A single value applies to both opening and extension.
            A tuple gives the respective affine penalties.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ClusterDatabase
            A handle resolving to the created cluster database.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> clusters = app.cluster_linear(database, threads=1).result()
        >>> isinstance(clusters, ClusterDatabase)
        True
        """
        database = ClusterDatabase(self, sequence_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                *_resolve_scoring(matrix, gap_penalty),
                CLIArgument(sequence_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="result2profile", allowed_options=_RESULT_TO_PROFILE_OPTIONS)
    def result_to_profile(
        self,
        alignment_db: AlignmentDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[ProfileDatabase[Any]]:
        """
        Compute a profile per query from search results using ``result2profile``.

        In contrast to the two-step route via :meth:`result_to_msa()` and
        :meth:`msa_to_profile()` this creates the profiles directly.

        Parameters
        ----------
        alignment_db : AlignmentDatabase
            The pairwise alignment results to summarize.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ProfileDatabase
            A handle resolving to the created profile database.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> hits = app.search(database, database, threads=1).result()
        >>> profiles = app.result_to_profile(hits, threads=1).result()
        >>> isinstance(profiles, ProfileDatabase)
        True
        """
        database = ProfileDatabase(self)
        return CommandSetup(
            parameters=[
                CLIArgument(alignment_db.query_db),
                CLIArgument(alignment_db.target_db),
                CLIArgument(alignment_db),
                CLIArgument(database),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(
        subcommand="profile2consensus", allowed_options=_PROFILE_TO_SEQUENCES_OPTIONS
    )
    def profile_to_consensus(
        self,
        profile_db: ProfileDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[SequenceDatabase[Any]]:
        """
        Extract the consensus sequence of each profile using
        ``profile2consensus``.

        Parameters
        ----------
        profile_db : ProfileDatabase
            The profile database to extract the sequences from.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of SequenceDatabase
            A handle resolving to the sequence database containing the consensus
            sequences.

        See Also
        --------
        profile_to_sequences
            Extracts the representative sequence instead, i.e. the first
            sequence of the underlying MSA.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> hits = app.search(database, database, threads=1).result()
        >>> profiles = app.result_to_profile(hits, threads=1).result()
        >>> consensus_db = app.profile_to_consensus(profiles, threads=1).result()
        >>> isinstance(consensus_db, SequenceDatabase)
        True
        """
        database = SequenceDatabase(self)
        return CommandSetup(
            parameters=[CLIArgument(profile_db), CLIArgument(database)],
            evaluate=lambda _out, _err: database,
        )

    @overload
    def profile_to_pssm(
        self,
        profile_db: ProfileDatabase[Any],
        output_path: None = None,
        **kwargs: Any,
    ) -> LocalProcessFuture[IO[bytes]]: ...

    @overload
    def profile_to_pssm(
        self,
        profile_db: ProfileDatabase[Any],
        output_path: PathLike[str] | str,
        **kwargs: Any,
    ) -> LocalProcessFuture[Path]: ...

    @command(subcommand="profile2pssm", allowed_options=_PROFILE_TO_PSSM_OPTIONS)
    def profile_to_pssm(
        self,
        profile_db: ProfileDatabase[Any],
        output_path: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[Path | IO[bytes]]:
        """
        Export profiles as position-specific scoring matrices using
        ``profile2pssm``.

        Parameters
        ----------
        profile_db : ProfileDatabase
            The profile database to export.
        output_path : path-like, optional
            The output path. If omitted, the output is written to a temporary
            binary file.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path or binary file
            A handle resolving to the output path, or to the temporary file if
            `output_path` is omitted.

        Notes
        -----
        Each profile is preceded by a line naming its database key and is
        followed by one row per profile column, giving the consensus residue and
        the integer score for each amino acid.

        Examples
        --------
        >>> app = MMseqsApp()
        >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
        >>> hits = app.search(database, database, threads=1).result()
        >>> profiles = app.result_to_profile(hits, threads=1).result()
        >>> pssm_file = app.profile_to_pssm(profiles, threads=1).result()
        >>> with open(pssm_file.name) as file:
        ...     for line in file.readlines()[:20]:
        ...         print(line, end="")
        Query profile of sequence 0
        Pos Cns A C D E F G H I K L M N P Q R S T V W Y
        0 M -2 -1 -3 -2 0 -3 -1 0 -2 1 7 -2 -2 0 -1 -3 -2 0 -1 -1
        1 K -1 -2 3 0 -3 -2 -1 -3 4 -2 -2 0 -1 0 0 -1 1 -2 -3 -2
        2 K -1 -3 -1 0 -3 -2 0 -3 5 -3 -2 0 -1 0 2 -1 -2 -3 -2 -2
        3 P -1 -2 -2 0 -3 -2 -1 -3 2 -3 -3 -2 8 0 0 -1 -2 -3 -3 -2
        4 Y -2 -2 -4 -2 3 -3 1 -2 -2 -1 -1 -3 -3 -2 -2 -3 -3 -2 2 8
        5 S 0 0 -1 0 -2 -1 -1 -2 -1 -2 -2 0 0 -1 -1 4 2 -2 -2 -2
        6 I -2 -1 -4 -4 0 -4 -3 5 -3 1 0 -4 -3 -3 -3 -3 -2 1 -2 -1
        7 G 0 -2 -2 -2 -3 6 -2 -4 -2 -4 -3 -1 -2 -2 -2 -1 -3 -4 -2 -3
        8 L -2 -1 -4 -3 0 -4 -3 0 -3 5 1 -4 -3 -2 -2 -3 -2 0 -1 -1
        9 D -2 -3 6 0 -3 -1 -1 -3 -1 -4 -3 0 -2 -1 -1 -1 -2 -4 -4 -3
        10 I -2 -1 -3 -4 0 -4 -3 5 -3 1 0 -4 -3 -3 -3 -3 -2 1 -2 -1
        11 G 0 -2 -2 -2 -3 6 -2 -4 -3 -4 -3 -1 -2 -2 -2 -1 -2 -4 -2 -3
        12 T 0 0 -1 -1 -2 -1 -2 -1 -1 -1 -1 0 -1 -1 -1 0 5 0 -2 -2
        13 N -2 -2 0 0 -3 0 0 -3 -1 -3 -3 6 -2 0 -1 0 -1 -3 -3 -2
        14 S 0 0 0 0 -2 0 -1 -3 -1 -3 -2 0 -1 0 -1 5 0 -2 -2 -2
        15 V -1 0 -3 -3 -1 -3 -3 1 -3 0 0 -3 -2 -3 -3 -2 -1 4 -2 -1
        16 G 0 -2 -1 -2 -3 6 -2 -4 -2 -4 -3 0 -2 -2 -2 -1 -2 -4 -2 -3
        17 W -3 -2 -5 -3 0 -3 -2 -3 -4 -2 -2 -4 -3 -2 -3 -3 -3 -3 12 1
        >>> pssm_file.close()
        """
        output: Path | IO[bytes]
        if output_path is None:
            output = NamedTemporaryFile("w+b")
        else:
            output = Path(output_path)
        return CommandSetup(
            parameters=[CLIArgument(profile_db), CLIArgument(output)],
            evaluate=lambda _out, _err: output,
        )


class FoldseekApp(MMseqsLikeApp):
    """
    A reusable handle to the *Foldseek* command line program.

    Parameters
    ----------
    path : path-like, optional
        Path to the ``foldseek`` executable.

    Attributes
    ----------
    tmp_dir : Path
        The parent directory for temporary command workspaces.

    Examples
    --------
    >>> structure_path = path_to_structures / "1aki.cif"
    >>> future = FoldseekApp().create_db(structure_path, threads=1)
    """

    _tmp_prefix = "foldseek_"

    def __init__(self, path: PathLike[str] | str = "foldseek") -> None:
        super().__init__(path)

    @command(subcommand="search", allowed_options=_STRUCTURE_SEARCH_OPTIONS)
    def search(
        self,
        query_db: SequenceDatabase[Any],
        target_db: SequenceDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Search a target database for structural matches using ``search``.

        Parameters
        ----------
        query_db, target_db : SequenceDatabase
            The query and target structure databases.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            search results.

        Examples
        --------
        >>> app = FoldseekApp()
        >>> database = app.create_db(path_to_structures / "1aki.cif").result()
        >>> alignment_db = app.search(database, database, threads=1).result()
        """
        database = AlignmentDatabase(self, query_db, target_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                CLIArgument(query_db),
                CLIArgument(target_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="structurealign", allowed_options=_STRUCTURE_ALIGN_OPTIONS)
    def structure_align(
        self,
        alignment_db: AlignmentDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Recompute structural alignments of a result database using
        ``structurealign``.

        This allows realigning the hits found by :meth:`search()` with different
        alignment parameters, without repeating the expensive prefilter stage.

        Parameters
        ----------
        alignment_db : AlignmentDatabase
            The result database whose hits are aligned.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            recomputed alignments.

        Examples
        --------
        >>> app = FoldseekApp()
        >>> database = app.create_db(path_to_structures / "1aki.cif").result()
        >>> hits = app.search(database, database, threads=1).result()
        >>> realigned = app.structure_align(hits, threads=1).result()
        >>> isinstance(realigned, AlignmentDatabase)
        True
        """
        database = AlignmentDatabase(
            self, alignment_db.query_db, alignment_db.target_db
        )
        return CommandSetup(
            parameters=[
                CLIArgument(alignment_db.query_db),
                CLIArgument(alignment_db.target_db),
                CLIArgument(alignment_db),
                CLIArgument(database),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="rbh", allowed_options=_STRUCTURE_SEARCH_OPTIONS)
    def search_reciprocal_best_hits(
        self,
        query_db: SequenceDatabase[Any],
        target_db: SequenceDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Find reciprocal best hits between two databases using ``rbh``.

        Only hits that are the best hit in both search directions are reported.

        Parameters
        ----------
        query_db, target_db : SequenceDatabase
            The query and target structure databases.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            reciprocal best hits.

        Examples
        --------
        >>> app = FoldseekApp()
        >>> database = app.create_db(path_to_structures / "1aki.cif").result()
        >>> hits = app.search_reciprocal_best_hits(database, database, threads=1)
        >>> isinstance(hits.result(), AlignmentDatabase)
        True
        """
        database = AlignmentDatabase(self, query_db, target_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                CLIArgument(query_db),
                CLIArgument(target_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="cluster", allowed_options=_STRUCTURE_CLUSTER_OPTIONS)
    def cluster(
        self,
        sequence_db: SequenceDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[ClusterDatabase[Any]]:
        """
        Cluster a structure database using the cascaded ``cluster`` workflow.

        Parameters
        ----------
        sequence_db : SequenceDatabase
            The structure database to cluster.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of ClusterDatabase
            A handle resolving to the created cluster database.

        Examples
        --------
        >>> app = FoldseekApp()
        >>> database = app.create_db(path_to_structures / "1aki.cif").result()
        >>> clusters = app.cluster(database, threads=1).result()
        >>> isinstance(clusters, ClusterDatabase)
        True
        """
        database = ClusterDatabase(self, sequence_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                CLIArgument(sequence_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @command(subcommand="multimersearch", allowed_options=_MULTIMER_SEARCH_OPTIONS)
    def search_multimers(
        self,
        query_db: SequenceDatabase[Any],
        target_db: SequenceDatabase[Any],
        **kwargs: Any,
    ) -> CommandSetup[AlignmentDatabase[Any]]:
        """
        Search multi-chain structures against each other using
        ``multimersearch``.

        In contrast to :meth:`search()` the hits are reported per complex instead
        of per chain.

        Parameters
        ----------
        query_db, target_db : SequenceDatabase
            The query and target structure databases.
            Chains of the same structure must share their structure identifier,
            which is the case for databases created from multi-chain files.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of AlignmentDatabase
            A handle resolving to the alignment database containing the
            complex-level hits.

        See Also
        --------
        create_multimer_report
            Reads the chain assignment of each complex-level hit.

        Examples
        --------
        >>> app = FoldseekApp()
        >>> database = app.create_db(path_to_structures / "1k6p.cif").result()
        >>> hits = app.search_multimers(database, database, threads=1).result()
        >>> isinstance(hits, AlignmentDatabase)
        True
        """
        database = AlignmentDatabase(self, query_db, target_db)
        work_dir = self._create_work_dir()
        return CommandSetup(
            parameters=[
                CLIArgument(query_db),
                CLIArgument(target_db),
                CLIArgument(database),
                CLIArgument(work_dir),
            ],
            evaluate=lambda _out, _err: database,
        )

    @overload
    def create_multimer_report(
        self,
        alignment_db: AlignmentDatabase[Any],
        output_path: None = None,
        **kwargs: Any,
    ) -> LocalProcessFuture[IO[bytes]]: ...

    @overload
    def create_multimer_report(
        self,
        alignment_db: AlignmentDatabase[Any],
        output_path: PathLike[str] | str,
        **kwargs: Any,
    ) -> LocalProcessFuture[Path]: ...

    @command(
        subcommand="createmultimerreport", allowed_options=_MULTIMER_REPORT_OPTIONS
    )
    def create_multimer_report(
        self,
        alignment_db: AlignmentDatabase[Any],
        output_path: PathLike[str] | str | None = None,
        **kwargs: Any,
    ) -> CommandSetup[Path | IO[bytes]]:
        """
        Export complex-level hits as a report using ``createmultimerreport``.

        Parameters
        ----------
        alignment_db : AlignmentDatabase
            The complex-level hits from :meth:`search_multimers()`.
        output_path : path-like, optional
            The output path. If omitted, the output is written to a temporary
            binary file.
        **kwargs
            Additional command line options.

        Returns
        -------
        future : Future of Path or binary file
            A handle resolving to the output path, or to the temporary file if
            `output_path` is omitted.

        Notes
        -----
        Each row gives the query and target identifier, the matched chains, the
        *TM-score* in both directions, the superimposition and the assignment ID.

        Examples
        --------
        >>> app = FoldseekApp()
        >>> database = app.create_db(path_to_structures / "1k6p.cif").result()
        >>> hits = app.search_multimers(database, database, threads=1).result()
        >>> report_file = app.create_multimer_report(hits, threads=1).result()
        >>> with open(report_file.name) as file:
        ...     query, target, query_chains, target_chains = file.readline().split()[:4]
        >>> print(query, target, query_chains, target_chains)
        1k6p 1k6p A,B A,B
        >>> report_file.close()
        """
        output: Path | IO[bytes]
        if output_path is None:
            output = NamedTemporaryFile("w+b")
        else:
            output = Path(output_path)
        return CommandSetup(
            parameters=[
                CLIArgument(alignment_db.query_db),
                CLIArgument(alignment_db.target_db),
                CLIArgument(alignment_db),
                CLIArgument(output),
            ],
            evaluate=lambda _out, _err: output,
        )


def _resolve_scoring(
    matrix: SubstitutionMatrix[Any, Any] | None,
    gap_penalty: int | tuple[int, int] | None,
) -> list[CLIOption]:
    """
    Turn a substitution matrix and a gap penalty into command line options.

    Parameters
    ----------
    matrix : SubstitutionMatrix or None
        The custom substitution matrix, which is written into a temporary file.
    gap_penalty : int or tuple of (int, int) or None
        The negative gap penalty.

    Returns
    -------
    options : list of CLIOption
        The options for the given scoring scheme, empty if neither is given.
    """
    options = []
    if matrix is not None:
        matrix_file = NamedTemporaryFile("w", suffix=".out")
        _write_substitution_matrix(matrix, matrix_file)
        matrix_file.flush()
        # The file is kept alive by the option it is referenced by
        options.append(CLIOption("sub_mat", matrix_file))
    gap_open, gap_extend = resolve_gap_penalty(gap_penalty)
    if gap_open is not None and gap_extend is not None:
        options += [
            CLIOption("gap_open", -gap_open),
            CLIOption("gap_extend", -gap_extend),
        ]
    return options


def _write_substitution_matrix(
    matrix: SubstitutionMatrix[Any, Any], file: IO[str]
) -> None:
    """
    Write a substitution matrix in the MMseqs2 matrix-file format.
    """
    if not matrix.is_symmetric():
        raise ValueError("MMseqs2 requires a symmetric substitution matrix")
    alphabet = matrix.get_alphabet1()
    include_mask = np.ones(len(alphabet), dtype=bool)
    if ProteinSequence.alphabet.extends(alphabet):
        # Ambiguous and stop amino-acid symbols are derived from canonical
        # residues and do not represent independent log-odds states.
        include_mask[ProteinSequence.alphabet.encode("B") :] = False
    indices = np.flatnonzero(include_mask)
    symbols = [
        str(matrix.get_alphabet1().decode(int(index))).upper() for index in indices
    ]
    if any(len(symbol) != 1 for symbol in symbols):
        raise ValueError("MMseqs2 matrix symbols must be single characters")
    if "X" in symbols or len(symbols) > 20:
        raise ValueError("MMseqs2 matrices support up to 20 defined symbols")
    active_scores = matrix.score_matrix()[np.ix_(indices, indices)]

    # MMseqs2 requires an explicit final row and column for its unknown state.
    scores = np.zeros((len(indices) + 1, len(indices) + 1), dtype=np.int32)
    scores[:-1, :-1] = active_scores
    original_symbols = [str(symbol).upper() for symbol in matrix.get_alphabet1()]
    if "X" in original_symbols:
        unknown_index = original_symbols.index("X")
        scores[-1, :-1] = matrix.score_matrix()[unknown_index, indices]
        scores[:-1, -1] = matrix.score_matrix()[indices, unknown_index]
        scores[-1, -1] = matrix.score_matrix()[unknown_index, unknown_index]
    symbols.append("X")

    file.write("# Generated by Biotite\n")
    file.write("    " + " ".join(f"{symbol:>3}" for symbol in symbols) + "\n")
    for symbol, row in zip(symbols, scores, strict=True):
        file.write(f"{symbol:>3} " + " ".join(f"{score:>3d}" for score in row) + "\n")
