from __future__ import annotations

__name__ = "biotite.application_v2.mmseqs"
__author__ = "Patrick Kunzmann"
__all__ = [
    "link_to_ss_database",
    "create_database_from_sequences",
    "create_database_from_msa",
    "create_database_from_structures",
    "get_sequences_from_database",
    "get_msa_from_database",
    "get_clusters",
    "get_matrix_from_profile",
    "get_multimer_report",
    "get_alignment_table",
    "get_alignments",
]

import csv
from collections.abc import Mapping
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import IO, Any, TypeVar, cast, overload
import numpy as np
from biotite.application_v2.mmseqs.app import (
    AlignmentFormatMode,
    DatabaseType,
    FoldseekApp,
    MMseqsApp,
    MMseqsLikeApp,
)
from biotite.application_v2.mmseqs.database import (
    AlignmentDatabase,
    ClusterDatabase,
    MSADatabase,
    ProfileDatabase,
    SequenceDatabase,
)
from biotite.sequence import (
    LetterAlphabet,
    NucleotideSequence,
    PositionalSequence,
    ProteinSequence,
    Sequence,
)
from biotite.sequence.align import (
    Alignment,
    SubstitutionMatrix,
    read_alignment_from_cigar,
)
from biotite.sequence.io.fasta import FastaFile, get_a3m_alignments
from biotite.structure import AtomArray
from biotite.structure.alphabet import I3DSequence
from biotite.structure.io.pdbx import CIFFile, set_structure

T = TypeVar("T", bound="MMseqsLikeApp")


def link_to_ss_database(app: T, database: SequenceDatabase[T]) -> SequenceDatabase[T]:
    """
    Link a Foldseek database to its 3Di sub-database.

    Parameters
    ----------
    app : MMseqsApp or FoldseekApp
        The application to use.
    database : SequenceDatabase
        The Foldseek database containing the 3Di sub-database.

    Returns
    -------
    linked_database : SequenceDatabase
        The linked 3Di database.
    """

    app.link_db(
        Path(f"{database.name}_h"),
        Path(f"{database.name}_ss_h"),
    ).result()
    ss_database = SequenceDatabase(app, database.path, suffix="_ss")
    ss_database.set_parent(database)
    return ss_database


@overload
def create_database_from_sequences(
    app: T,
    sequences: Mapping[str, ProteinSequence | str],
    prostt5_weights: SequenceDatabase[T] | None = None,
) -> SequenceDatabase[T]: ...


@overload
def create_database_from_sequences(
    app: MMseqsApp,
    sequences: Mapping[str, NucleotideSequence],
) -> SequenceDatabase[MMseqsApp]: ...


def create_database_from_sequences(
    app: MMseqsLikeApp,
    sequences: Mapping[str, Sequence | str],
    prostt5_weights: SequenceDatabase[Any] | None = None,
) -> SequenceDatabase[Any]:
    """
    Create a database from sequences.

    The molecule type is taken from the type of the given sequences, so that
    short sequences are not misinterpreted by the detection of the application.
    Plain strings are interpreted as protein sequences.

    Parameters
    ----------
    app : MMseqsApp or FoldseekApp
        The application to use.
        Nucleotide sequences are only supported by :class:`MMseqsApp`, as
        *Foldseek* compares structures.
    sequences : mapping of str to Sequence or str
        Sequence identifiers mapped to the respective sequence.
        All sequences must be of the same type.
        For *Foldseek* the corresponding *3Di* sequences are predicted using
        *ProstT5*.
    prostt5_weights : SequenceDatabase, optional
        ProstT5 weights for Foldseek. If omitted, they are downloaded.

    Returns
    -------
    database : SequenceDatabase
        The created database.

    Notes
    -----
    A search on a nucleotide database requires the ``search_type`` option (see
    :class:`SearchType`), as the application cannot infer whether the sequences
    should be aligned directly or translated beforehand.
    """
    db_type = _resolve_database_type(sequences)
    # The file is closed after writing, so that the application can open it on
    # Windows as well, and it is deleted when the context is exited
    with NamedTemporaryFile("w+", suffix=".fasta", delete_on_close=False) as temp_file:
        _write_fasta(temp_file, sequences)
        temp_file.close()
        match app:
            case MMseqsApp():
                return app.create_db([temp_file], db_type=db_type).result()
            case FoldseekApp():
                if db_type != DatabaseType.AMINO_ACID:
                    raise TypeError(
                        "Foldseek only supports protein sequences, "
                        f"but got '{db_type.name}'"
                    )
                if prostt5_weights is None:
                    prostt5_weights = app.databases("ProstT5").result()
                return app.create_db(
                    [temp_file], prostt5_model=prostt5_weights
                ).result()
            case _:
                raise TypeError(f"Unsupported application: '{type(app).__name__}'")


def create_database_from_structures(
    app: FoldseekApp,
    structures: Mapping[str, AtomArray],
    **kwargs: Any,
) -> SequenceDatabase[FoldseekApp]:
    """
    Create a *Foldseek* database from structures.

    Parameters
    ----------
    app : FoldseekApp
        The application to use.
    structures : mapping of str to AtomArray
        Structure identifiers mapped to the respective structure.
        Identifiers must not contain whitespace or a path separator.
        Missing ``b_factor`` and ``occupancy`` annotations are filled with
        default values, as *Foldseek* requires those columns.
    **kwargs
        Additional command line options passed to ``createdb``.

    Returns
    -------
    database : SequenceDatabase
        The created 3Di database.
        Each chain becomes a separate entry, identified by the structure
        identifier and the chain ID, separated by an underscore.
    """
    for identifier in structures:
        if any(character.isspace() for character in identifier):
            raise ValueError(f"Identifier '{identifier}' must not contain whitespace")
        if "/" in identifier:
            raise ValueError(f"Identifier '{identifier}' must not contain '/'")
    with TemporaryDirectory() as directory:
        paths = []
        for identifier, atoms in structures.items():
            # *Foldseek* requires these columns in the written file
            defaults = {"b_factor": 0.0, "occupancy": 1.0}
            missing = defaults.keys() - set(atoms.get_annotation_categories())
            if missing:
                atoms = atoms.copy()
                for category in missing:
                    atoms.set_annotation(
                        category,
                        np.full(atoms.array_length(), defaults[category]),
                    )
            path = Path(directory) / f"{identifier}.cif"
            cif_file = CIFFile()
            set_structure(cif_file, atoms)
            cif_file.write(path)
            paths.append(path)
        # The command copies the structures into the database,
        # hence the temporary files are not required anymore afterwards
        return app.create_db(paths, chain_name_mode=1, **kwargs).result()


def create_database_from_msa(
    app: MMseqsApp,
    msa: Mapping[str, list[Alignment]],
) -> MSADatabase[MMseqsApp]:
    """
    Create an MSA database from one or multiple MSAs.

    Parameters
    ----------
    app : MMseqsApp
        The application to use.
    msa : dict (str -> list of Alignment)
        MSA identifiers mapped to the respective MSA.
        Each MSA is given as pairwise alignments of its query to each of its
        homologs.
        Hence, the first sequence must be the same query sequence in each
        alignment of an MSA.

    Returns
    -------
    database : MSADatabase
        The created database, containing one MSA per given identifier.

    Raises
    ------
    ValueError
        If an identifier in ``msa`` contains whitespace.

    See Also
    --------
    get_msa_from_database : The inverse operation.

    Notes
    -----
    The identifiers appear as ``query`` in the alignment table of a search,
    which allows mapping the hits back to the respective MSA.

    Each alignment is projected onto the ungapped query sequence, i.e.
    insertions with respect to the query are removed.
    Hence, a profile created from the returned database via
    :meth:`MMseqsApp.msa_to_profile()` is in the same coordinate frame as the
    query.
    """
    for identifier in msa:
        if any(character.isspace() for character in identifier):
            raise ValueError(f"Identifier '{identifier}' must not contain whitespace")
    # See `create_database_from_sequences()` for the handling of the temporary file
    with NamedTemporaryFile("w+", suffix=".sto", delete_on_close=False) as temp_file:
        for identifier, alignments in msa.items():
            _write_stockholm(temp_file, alignments, identifier)
        temp_file.close()
        return app.convert_msa(temp_file).result()


@overload
def get_sequences_from_database(
    app: T,
    sequence_db: SequenceDatabase[T],
    as_3di: bool = False,
) -> dict[str, str]: ...


@overload
def get_sequences_from_database(
    app: MMseqsApp,
    sequence_db: ProfileDatabase[MMseqsApp],
    as_3di: bool = False,
) -> dict[str, str]: ...


def get_sequences_from_database(
    app: MMseqsLikeApp,
    sequence_db: SequenceDatabase[Any] | ProfileDatabase[Any],
    as_3di: bool = False,
) -> dict[str, str]:
    """
    Extract the sequences from a database.

    Parameters
    ----------
    app : MMseqsApp or FoldseekApp
        The application to use.
    sequence_db : SequenceDatabase or ProfileDatabase
        The database to read.
        For a profile database, the representative sequence of each profile is
        returned, which is the query sequence for profiles created from MSAs.
        Profile databases are only supported by :class:`MMseqsApp`.
    as_3di : bool, optional
        If true, extract Foldseek 3Di instead of amino-acid sequences.

    Returns
    -------
    sequences : dict of str to str
        Sequence identifiers mapped to sequences.
    """
    exportable_db: SequenceDatabase[Any]
    if isinstance(sequence_db, ProfileDatabase):
        # A profile cannot be exported as FASTA, but its representative sequence can
        # The overloads ensure that only an `MMseqsApp` is given for profiles
        exportable_db = cast(MMseqsApp, app).profile_to_sequences(sequence_db).result()
    else:
        exportable_db = sequence_db
    if as_3di:
        exportable_db = link_to_ss_database(app, exportable_db)
    file = app.convert_to_fasta(exportable_db).result()
    try:
        # The application writes through a separate file descriptor, so the
        # retained temporary file's read buffer may still cache the original
        # empty file. Reopening its path observes the application output.
        with open(file.name, "rb") as input_file:
            content = input_file.read().decode()
    finally:
        file.close()
    with StringIO(content) as text:
        return {
            _identifier(header): sequence
            for header, sequence in FastaFile.read_iter(text)
        }


def get_msa_from_database(
    app: MMseqsApp,
    msa_db: MSADatabase[MMseqsApp],
    sequence_type: type[Sequence] = ProteinSequence,
) -> dict[str, list[Alignment]]:
    """
    Extract the MSAs from an MSA database.

    Parameters
    ----------
    app : MMseqsApp
        The application to use.
    msa_db : MSADatabase
        The database to read.
        Its entries must be in *A3M* format, i.e. the first sequence of each MSA
        must be an ungapped query.
    sequence_type : type of Sequence, optional
        The type the aligned sequences are parsed into.

    Returns
    -------
    msa : dict (str -> list of Alignment)
        MSA identifiers mapped to the respective MSA, given as pairwise
        alignments of its query to each of its homologs.

    See Also
    --------
    create_database_from_msa : The inverse operation.

    Notes
    -----
    The identifier of each MSA is taken from the first sequence of the MSA, as
    an MSA database does not store identifiers separately.
    """
    msa = {}
    with TemporaryDirectory() as directory:
        for path in sorted(app.unpack_db(msa_db, directory).result().iterdir()):
            fasta_file = FastaFile.read(path)
            identifier = _identifier(next(iter(fasta_file.keys())))
            msa[identifier] = get_a3m_alignments(fasta_file, sequence_type)
    return msa


def get_clusters(
    app: T,
    cluster_db: ClusterDatabase[T],
) -> dict[str, list[str]]:
    """
    Extract the clusters from a cluster database.

    Parameters
    ----------
    app : MMseqsApp or FoldseekApp
        The application to use.
    cluster_db : ClusterDatabase
        The database to read.

    Returns
    -------
    clusters : dict (str -> list of str)
        The identifier of each representative sequence mapped to the
        identifiers of its cluster members.
        The representative is the first member of its own cluster.
    """
    clusters: dict[str, list[str]] = {}
    for representative, member in _read_tsv(app.create_tsv(cluster_db)):
        clusters.setdefault(representative, []).append(member)
    return clusters


def get_multimer_report(
    app: FoldseekApp,
    alignment_db: AlignmentDatabase[FoldseekApp],
) -> dict[str, list[str]]:
    """
    Extract the complex-level hits from a multimer search.

    Parameters
    ----------
    app : FoldseekApp
        The application to use.
    alignment_db : AlignmentDatabase
        The complex-level hits from :meth:`FoldseekApp.search_multimers()`.

    Returns
    -------
    report : dict (str -> list of str)
        The report columns mapped to their raw string values.
        ``query`` and ``target`` give the structure identifiers, ``query_chains``
        and ``target_chains`` the comma-separated matched chains,
        ``query_tm_score`` and ``target_tm_score`` the *TM-score* in the
        respective direction, and ``rotation`` and ``translation`` the
        superimposition of the target onto the query.
    """
    columns = [
        "query",
        "target",
        "query_chains",
        "target_chains",
        "query_tm_score",
        "target_tm_score",
        "rotation",
        "translation",
    ]
    report: dict[str, list[str]] = {column: [] for column in columns}
    for row in _read_tsv(app.create_multimer_report(alignment_db)):
        for column, value in zip(columns, row, strict=False):
            report[column].append(value)
    return report


def get_matrix_from_profile(
    app: MMseqsApp,
    profile_db: ProfileDatabase[MMseqsApp],
) -> list[SubstitutionMatrix[Any, Any]]:
    """
    Get a position-specific scoring matrix for each profile in a database.

    Parameters
    ----------
    app : MMseqsApp
        The application to use.
    profile_db : ProfileDatabase
        The database to read.

    Returns
    -------
    matrices : list of SubstitutionMatrix
        One matrix per profile, in the order the profiles appear in the
        database.
        The first alphabet is the alphabet of a :class:`PositionalSequence` over
        the consensus sequence of the profile, the second one is the standard
        protein alphabet.

    Notes
    -----
    *MMseqs2* scores only the 20 canonical amino acids, hence the columns of the
    ambiguous and stop symbols are zero.
    For the same reason only profiles of protein sequences are supported:
    the application emits amino acid columns even for a nucleotide profile.

    Examples
    --------

    >>> app = MMseqsApp()
    >>> database = app.create_db(path_to_sequences / "cas9.fasta").result()
    >>> hits = app.search(database, database, threads=1).result()
    >>> profiles = app.result_to_profile(hits, threads=1).result()
    >>> matrix = get_matrix_from_profile(app, profiles)[0]
    >>> print(matrix.shape)
    (1368, 24)
    >>> # One symbol per profile column, shown as the consensus residue
    >>> print([str(symbol) for symbol in matrix.get_alphabet1().get_symbols()[:10]])
    ['M', 'K', 'K', 'P', 'Y', 'S', 'I', 'G', 'L', 'D']
    >>> # The protein alphabet
    >>> print(matrix.get_alphabet2())
    ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',
     'T', 'V', 'W', 'Y', 'B', 'Z', 'X', '*')
    """
    pssm_file = app.profile_to_pssm(profile_db).result()
    try:
        # See `get_sequences_from_database()` for why the temporary path is
        # reopened here
        with open(pssm_file.name, "rb") as input_file:
            content = input_file.read().decode()
    finally:
        pssm_file.close()

    alphabet = ProteinSequence.alphabet
    matrices = []
    consensus: list[str] = []
    scores: list[list[int]] = []
    column_indices: list[int] = []

    def finalize() -> None:
        if not consensus:
            return
        score_matrix = np.zeros((len(consensus), len(alphabet)), dtype=np.int32)
        score_matrix[:, column_indices] = np.array(scores, dtype=np.int32)
        positional = PositionalSequence(ProteinSequence("".join(consensus)))
        matrices.append(SubstitutionMatrix(positional.alphabet, alphabet, score_matrix))

    for line in content.splitlines():
        if not line:
            continue
        if line.startswith("Query profile"):
            finalize()
            consensus, scores = [], []
        elif line.startswith("Pos"):
            column_indices = [
                alphabet.encode(symbol) for symbol in line.split("\t")[2:]
            ]
        else:
            fields = line.split("\t")
            consensus.append(fields[1])
            scores.append([int(field) for field in fields[2:]])
    finalize()
    return matrices


def get_alignment_table(
    app: T,
    alignment_db: AlignmentDatabase[T],
    columns: list[str],
) -> dict[str, list[str]]:
    """
    Export selected alignment columns.

    Parameters
    ----------
    app : MMseqsApp or FoldseekApp
        The application to use.
    alignment_db : AlignmentDatabase
        The alignment database to read.
    columns : list of str
        The ``convertalis --format-output`` columns to include.

    Returns
    -------
    table : dict of str to list of str
        Columns mapped to their raw string values.
    """
    if len(columns) == 0:
        raise ValueError("At least one alignment column is required")
    if len(set(columns)) != len(columns):
        raise ValueError("Alignment columns must be unique")

    file = app.convert_alignments(
        alignment_db,
        format_mode=AlignmentFormatMode.BLAST_TABLE_WITH_HEADERS,
        format_output=",".join(columns),
    ).result()
    try:
        # See `get_sequences_from_database()` for why the temporary path is
        # reopened here
        with open(file.name, "rb") as input_file:
            content = input_file.read().decode()
    finally:
        file.close()
    table: dict[str, list[str]] = {column: [] for column in columns}
    with StringIO(content, newline="") as text:
        for row in csv.DictReader(text, delimiter="\t"):
            for column in columns:
                table[column].append(row[column])
    return table


def get_alignments(
    app: T,
    alignment_db: AlignmentDatabase[T],
    sequence_type: type[Sequence] = ProteinSequence,
) -> dict[tuple[str, str], Alignment]:
    """
    Reconstruct Biotite alignments from an alignment database.

    Parameters
    ----------
    app : MMseqsApp or FoldseekApp
        The application to use.
    alignment_db : AlignmentDatabase
        The alignment database to read.
        The search that created `alignment_db` must enable backtracing with ``a=True``.
    sequence_type : type of Sequence, optional
        The type the aligned sequences are parsed into.
        :class:`biotite.structure.alphabet.I3DSequence` reconstructs the
        alignments from the *3Di* sequences of a *Foldseek* database instead of
        from the amino acid sequences.

    Returns
    -------
    alignments : dict
        ``(query_id, target_id)`` pairs mapped to alignments. Entries are ordered
        by increasing E-value.

    Notes
    -----
    The identifiers are the ones the application parsed from the FASTA headers,
    e.g. the accession for *UniProt*-style headers, and thus may differ from the
    identifiers given to :func:`get_sequences_from_database()`.
    """
    as_3di = sequence_type is I3DSequence
    query_db = alignment_db.query_db
    if isinstance(query_db, ProfileDatabase):
        # Only `MMseqsApp` supports profiles, so `app` must be one if the search
        # was run with a profile database as query
        query_sequences = get_sequences_from_database(
            cast(MMseqsApp, app), cast(ProfileDatabase[MMseqsApp], query_db), as_3di
        )
    else:
        query_sequences = get_sequences_from_database(app, query_db, as_3di)
    columns = ["query", "target", "qheader", "qstart", "tstart", "cigar", "evalue"]
    if as_3di:
        # The alignment table only provides the amino acid sequences,
        # hence the *3Di* sequences need to be read from the database
        table = get_alignment_table(app, alignment_db, [*columns, "theader"])
        target_sequences = get_sequences_from_database(
            app, alignment_db.target_db, as_3di=True
        )
        table["tseq"] = [
            target_sequences[_identifier(header)] for header in table["theader"]
        ]
    else:
        # Let the application report the sequences of the hits,
        # which avoids reading the entire target database
        table = get_alignment_table(app, alignment_db, [*columns, "tseq"])
    row_indices = sorted(
        range(len(table["query"])),
        key=lambda index: float(table["evalue"][index]),
    )
    alignments = {}
    for index in row_indices:
        query_id = table["query"][index]
        target_id = table["target"][index]
        query = sequence_type(query_sequences[_identifier(table["qheader"][index])])
        target = sequence_type(table["tseq"][index])
        query_start = int(table["qstart"][index]) - 1
        target_start = int(table["tstart"][index]) - 1
        alignment = read_alignment_from_cigar(table["cigar"][index], 0, target, query)[
            :, ::-1
        ]
        alignment.trace[alignment.trace[:, 0] != -1, 0] += query_start
        alignment.trace[alignment.trace[:, 1] != -1, 1] += target_start
        alignments[(query_id, target_id)] = alignment
    return alignments


def _identifier(header: str) -> str:
    """
    Get the identifier of a FASTA header, i.e. its first word.

    Parameters
    ----------
    header : str
        The FASTA header without the leading ``>``.

    Returns
    -------
    identifier : str
        The first whitespace-separated word of the header.
    """
    words = header.split(maxsplit=1)
    return words[0] if words else ""


def _write_stockholm(
    file: IO[str],
    alignments: list[Alignment],
    identifier: str,
) -> None:
    """
    Write the given alignments as MSA into a *Stockholm* file.

    Parameters
    ----------
    file : text file
        The file to write to.
    alignments : list of Alignment
        The pairwise alignments of the query to each of its homologs.
        The first sequence must be the same query sequence in each alignment.
    identifier : str
        The identifier the written MSA gets in an MSA database.

    Notes
    -----
    Each alignment is projected onto the ungapped query sequence, as
    *Stockholm* has no notion of insertions with respect to the query.
    """
    if len(alignments) == 0:
        raise ValueError("The MSA must contain at least one alignment")

    query_sequence = alignments[0].sequences[0]
    aligned_sequences = [str(query_sequence)]
    for alignment in alignments:
        # Usually all alignments share the same query object,
        # which makes comparing the sequence codes unnecessary
        if alignment.sequences[0] is not query_sequence and not np.array_equal(
            alignment.sequences[0].code, query_sequence.code
        ):
            raise ValueError("All alignments must have the same query sequence")
        aligned_sequences.append(_project_onto_query(alignment))

    file.write("# STOCKHOLM 1.0\n")
    # Without this comment 'convertmsa' would fall back to the name of the
    # first sequence, which would be the same for every MSA in the file
    file.write(f"#=GF AC {identifier}\n")
    for i, sequence in enumerate(aligned_sequences):
        file.write(f"seq{i:07d} {sequence}\n")
    file.write("//\n")


def _project_onto_query(alignment: Alignment) -> str:
    """
    Project the hit of a pairwise alignment onto the ungapped query sequence.

    Parameters
    ----------
    alignment : Alignment
        The alignment of the query to one of its hits.
        The query must be the first sequence.

    Returns
    -------
    projection : str
        The hit sequence, with insertions with respect to the query removed and
        a gap symbol wherever the hit has no residue.
        It has the length of the query, i.e. query positions the alignment does
        not cover, e.g. the termini of a local alignment, are gaps as well.
    """
    query_sequence, hit_sequence = alignment.sequences
    alphabet = hit_sequence.alphabet
    if not isinstance(alphabet, LetterAlphabet):
        raise TypeError(
            "The hit sequence must have a letter alphabet to be written as text"
        )
    trace = alignment.trace
    # *Stockholm* has no notion of insertions with respect to the query
    # -> keep only the columns where the query has a residue
    is_query_column = trace[:, 0] != -1
    query_indices = trace[is_query_column, 0]
    hit_indices = trace[is_query_column, 1]
    # Start from gaps over the entire query, as the alignment may not cover all
    # of it
    symbols = np.full(len(query_sequence), b"-", dtype="S1")
    is_present = hit_indices != -1
    symbols[query_indices[is_present]] = alphabet.decode_multiple_into_bytes(
        hit_sequence.code[hit_indices[is_present]]
    )
    return symbols.tobytes().decode("ascii")


def _read_tsv(future: Any) -> list[list[str]]:
    """
    Read the tab-separated output file a command resolves to.

    Parameters
    ----------
    future : Future of binary file
        The pending command writing the file.

    Returns
    -------
    rows : list of list of str
        The fields of each row.
    """
    file = future.result()
    try:
        # See `get_sequences_from_database()` for why the temporary path is
        # reopened here
        with open(file.name, "rb") as input_file:
            content = input_file.read().decode()
    finally:
        file.close()
    return [line.split("\t") for line in content.splitlines() if line]


def _write_fasta(file: IO[str], sequences: Mapping[str, str | Sequence]) -> None:
    """
    Write sequences into a FASTA file and flush it.

    Parameters
    ----------
    file : text file
        The file to write to.
    sequences : mapping of str to str or Sequence
        Sequence identifiers mapped to the respective sequence.
    """
    FastaFile.write_iter(
        file,
        ((identifier, str(sequence)) for identifier, sequence in sequences.items()),
        # Avoid line breaks within a sequence, as they are not needed here
        chars_per_line=1_000_000,
    )
    file.flush()


def _resolve_database_type(
    sequences: Mapping[str, Sequence | str],
) -> DatabaseType:
    """
    Determine the database type from the type of the given sequences.

    Parameters
    ----------
    sequences : mapping of str to Sequence or str
        The sequences the database is created from.

    Returns
    -------
    db_type : DatabaseType
        The type matching the given sequences.

    Raises
    ------
    ValueError
        If the sequences are not all of the same type.
    """
    is_nucleotide = {
        isinstance(sequence, NucleotideSequence) for sequence in sequences.values()
    }
    if len(is_nucleotide) > 1:
        raise ValueError("All sequences must be of the same type")
    if is_nucleotide == {True}:
        return DatabaseType.NUCLEOTIDE
    return DatabaseType.AMINO_ACID
