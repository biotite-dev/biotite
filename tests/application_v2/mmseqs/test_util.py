# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from pathlib import Path
import numpy as np
import pytest
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.io.fasta as fasta
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx
from biotite.application_v2.mmseqs import (
    ProfileDatabase,
    create_database_from_msa,
    create_database_from_sequences,
    create_database_from_structures,
    get_alignment_table,
    get_alignments,
    get_clusters,
    get_matrix_from_profile,
    get_msa_from_database,
    get_multimer_report,
    get_sequences_from_database,
)

PROTEIN_ALPHABET = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
NUCLEOTIDE_ALPHABET = np.array(list("ACGT"))


def _random_protein_sequence(rng, length=None):
    if length is None:
        length = int(rng.integers(35, 61))
    return "".join(rng.choice(PROTEIN_ALPHABET, size=length))


def _mutate_sequence(rng, sequence, rate):
    """
    Randomize each position of the given sequence string with probability `rate`.
    """
    symbols = np.array(list(sequence))
    mutation_mask = rng.random(len(symbols)) < rate
    symbols[mutation_mask] = rng.choice(
        PROTEIN_ALPHABET, size=np.count_nonzero(mutation_mask)
    )
    return "".join(symbols)


def _random_msa(rng, query, n_homologs, mutation_rate):
    """
    Create an MSA of a query with mutated copies of itself, given as pairwise
    gapless alignments.
    """
    indices = np.arange(len(query))
    trace = np.stack([indices, indices], axis=-1)
    return [
        align.Alignment(
            [
                query,
                seq.ProteinSequence(_mutate_sequence(rng, str(query), mutation_rate)),
            ],
            trace,
        )
        for _ in range(n_homologs)
    ]


def _random_sequence_mapping(rng):
    count = int(rng.integers(1, 5))
    return {
        f"sequence_{index}": _random_protein_sequence(rng) for index in range(count)
    }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("nucleotide", [False, True])
def test_sequence_roundtrip_properties(mmseqs_app, nucleotide, seed):
    """
    Arbitrary valid protein and nucleotide mappings survive a database roundtrip
    unchanged.
    """
    rng = np.random.default_rng(seed)
    if nucleotide:
        # The molecule type is taken from the sequence type
        sequences = {
            f"sequence_{index}": seq.NucleotideSequence(
                "".join(rng.choice(NUCLEOTIDE_ALPHABET, size=int(rng.integers(35, 61))))
            )
            for index in range(int(rng.integers(1, 5)))
        }
    else:
        sequences = _random_sequence_mapping(rng)
    database = create_database_from_sequences(mmseqs_app, sequences)

    assert get_sequences_from_database(mmseqs_app, database) == {
        identifier: str(sequence) for identifier, sequence in sequences.items()
    }


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    ["header", "identifier"],
    [
        ("query", "query"),
        # MMseqs2 parses the accession from UniProt-style headers,
        # so the reported identifier differs from the first word of the header
        ("sp|P12345|QUERY_HUMAN Some description", "P12345"),
    ],
)
def test_alignment_result_properties(mmseqs_app, seed, header, identifier):
    """
    Self-search tables are rectangular and reconstruct identity alignments,
    also if the application parses the identifier from the header.
    """
    sequence = _random_protein_sequence(np.random.default_rng(seed))
    database = create_database_from_sequences(
        mmseqs_app, {header: seq.ProteinSequence(sequence)}
    )
    alignment_db = mmseqs_app.search(
        database,
        database,
        a=True,
        k=5,
        e=10,
        threads=1,
    ).result()

    columns = ["query", "target", "qstart", "tstart", "cigar", "evalue"]
    table = get_alignment_table(mmseqs_app, alignment_db, columns)
    assert list(table) == columns
    assert len({len(values) for values in table.values()}) == 1
    assert set(table["query"]) == {identifier}
    assert set(table["target"]) == {identifier}

    alignments = get_alignments(mmseqs_app, alignment_db)
    assert list(alignments) == [(identifier, identifier)]
    alignment = alignments[(identifier, identifier)]
    assert alignment.sequences[0] == alignment.sequences[1]
    assert str(alignment).splitlines()[0] == str(alignment).splitlines()[1]


@pytest.mark.parametrize("as_3di", [False, True])
def test_foldseek_sequence_properties(foldseek_app, structure_path, as_3di):
    """
    Foldseek extraction agrees with Biotite for amino-acid and 3Di sequences.
    """
    atoms = pdbx.get_structure(pdbx.CIFFile.read(structure_path), model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if as_3di:
        reference_sequences = strucalph.to_3di(atoms)
    else:
        reference_sequences = struc.to_sequence(atoms)
    reference = {
        f"1aki_{atoms.chain_id[start]}": str(sequence).upper()
        for sequence, start in zip(*reference_sequences)
    }

    database = foldseek_app.create_db(
        structure_path, chain_name_mode=1, threads=1
    ).result()
    sequences = get_sequences_from_database(foldseek_app, database, as_3di=as_3di)

    assert sequences == reference


@pytest.mark.parametrize("columns", [[], ["query", "query"]])
def test_invalid_alignment_columns(mmseqs_app, columns):
    """
    Empty or duplicate table schemas are rejected without launching a process.
    """
    with pytest.raises(ValueError):
        get_alignment_table(mmseqs_app, None, columns)


@pytest.mark.parametrize("seed", range(3))
def test_profile_search(mmseqs_app, seed):
    """
    A profile built from an MSA finds a member of the same family, but not an
    unrelated sequence, and its representative sequence is the MSA query.
    """
    HOMOLOG_ID = "homolog"
    UNRELATED_ID = "unrelated"
    SEQUENCE_LENGTH = 100
    MUTATION_RATE = 0.1

    rng = np.random.default_rng(seed)
    query = seq.ProteinSequence(_random_protein_sequence(rng, SEQUENCE_LENGTH))
    msa = _random_msa(rng, query, 4, MUTATION_RATE)
    msa_db = create_database_from_msa(mmseqs_app, {"query": msa})
    profile_db = mmseqs_app.msa_to_profile(msa_db, threads=1).result()
    assert isinstance(profile_db, ProfileDatabase)
    assert get_sequences_from_database(mmseqs_app, profile_db) == {"query": str(query)}

    target_db = create_database_from_sequences(
        mmseqs_app,
        {
            HOMOLOG_ID: _mutate_sequence(rng, str(query), MUTATION_RATE),
            UNRELATED_ID: _random_protein_sequence(rng, SEQUENCE_LENGTH),
        },
    )
    # Keep k-mer size low to avoid long creation times of k-mer pointer table
    alignment_db = mmseqs_app.search(
        profile_db, target_db, k=5, a=True, threads=1
    ).result()
    alignment_table = get_alignment_table(mmseqs_app, alignment_db, ["query", "target"])
    alignments = get_alignments(mmseqs_app, alignment_db)

    assert set(alignment_table["target"]) == {HOMOLOG_ID}
    # The alignment is reconstructed against the representative sequence of the
    # profile, which is the query of the MSA
    assert list(alignments.keys()) == [("query", HOMOLOG_ID)]
    assert alignments["query", HOMOLOG_ID].sequences[0] == query


@pytest.mark.parametrize("seed", range(3))
def test_batched_profile_search(mmseqs_app, seed):
    """
    Multiple MSAs in one database give hits attributable to the respective MSA.
    """
    N_QUERIES = 3
    SEQUENCE_LENGTH = 100
    MUTATION_RATE = 0.1

    rng = np.random.default_rng(seed)
    queries = {
        f"query_{i}": seq.ProteinSequence(
            _random_protein_sequence(rng, SEQUENCE_LENGTH)
        )
        for i in range(N_QUERIES)
    }
    msas = {
        identifier: _random_msa(rng, query, 4, MUTATION_RATE)
        for identifier, query in queries.items()
    }
    profile_db = mmseqs_app.msa_to_profile(
        create_database_from_msa(mmseqs_app, msas), threads=1
    ).result()
    target_db = create_database_from_sequences(
        mmseqs_app,
        {
            f"target_{i}": _mutate_sequence(rng, str(query), MUTATION_RATE)
            for i, query in enumerate(queries.values())
        },
    )

    alignment_db = mmseqs_app.search(profile_db, target_db, k=5, threads=1).result()
    alignment_table = get_alignment_table(mmseqs_app, alignment_db, ["query", "target"])

    # Each query must find its own target and no other
    hits = dict(zip(alignment_table["query"], alignment_table["target"]))
    assert hits == {f"query_{i}": f"target_{i}" for i in range(N_QUERIES)}


def test_profile_representative_roundtrip(mmseqs_app):
    """
    A roundtrip of a real MSA through an MSA database, a profile database and the
    representative sequence database recovers the query sequence of the MSA.
    """
    a3m_file = fasta.FastaFile.read(Path("tests/sequence/data/1a00_A_uniref90.a3m"))
    msa = fasta.get_a3m_alignments(a3m_file)
    query = msa[0].sequences[0]

    msa_db = create_database_from_msa(mmseqs_app, {"query": msa})
    profile_db = mmseqs_app.msa_to_profile(msa_db, threads=1).result()
    representative_db = mmseqs_app.profile_to_sequences(profile_db, threads=1).result()

    assert get_sequences_from_database(mmseqs_app, representative_db) == {
        "query": str(query)
    }


@pytest.mark.parametrize("seed", range(3))
def test_profile_from_msa_with_insertions(mmseqs_app, seed):
    """
    Insertions with respect to the query do not become columns of the profile.
    """
    SEQUENCE_LENGTH = 100
    MUTATION_RATE = 0.1
    POSITION = 50
    INSERTION_LENGTH = 5

    rng = np.random.default_rng(seed)
    query = seq.ProteinSequence(_random_protein_sequence(rng, SEQUENCE_LENGTH))
    # The query has gaps where the homologs have an insertion
    gapped_query = (
        str(query)[:POSITION] + "-" * INSERTION_LENGTH + str(query)[POSITION:]
    )
    msa = []
    for _ in range(4):
        member = _mutate_sequence(rng, str(query), MUTATION_RATE)
        insertion = _random_protein_sequence(rng, INSERTION_LENGTH)
        gapped_member = member[:POSITION] + insertion + member[POSITION:]
        msa.append(
            align.Alignment(
                [query, seq.ProteinSequence(gapped_member)],
                align.Alignment.trace_from_strings([gapped_query, gapped_member]),
            )
        )

    profile_db = mmseqs_app.msa_to_profile(
        create_database_from_msa(mmseqs_app, {"query": msa}), threads=1
    ).result()
    target_db = create_database_from_sequences(mmseqs_app, {"target": query})
    alignment_db = mmseqs_app.search(profile_db, target_db, k=5, threads=1).result()
    alignment_table = get_alignment_table(
        mmseqs_app, alignment_db, ["query", "target", "qlen"]
    )

    # The profile length is the ungapped query length, not the aligned length
    assert alignment_table["qlen"] == [str(SEQUENCE_LENGTH)]


@pytest.mark.parametrize("seed", range(3))
def test_clusters(mmseqs_app, seed):
    """
    Clustering families of mutated sequences recovers the families.
    """
    N_FAMILIES = 3
    N_MEMBERS = 3
    SEQUENCE_LENGTH = 100
    MUTATION_RATE = 0.05

    rng = np.random.default_rng(seed)
    families = {}
    sequences = {}
    for family in range(N_FAMILIES):
        ancestor = _random_protein_sequence(rng, SEQUENCE_LENGTH)
        identifiers = []
        for member in range(N_MEMBERS):
            identifier = f"family_{family}_member_{member}"
            sequences[identifier] = _mutate_sequence(rng, ancestor, MUTATION_RATE)
            identifiers.append(identifier)
        families[family] = set(identifiers)

    database = create_database_from_sequences(mmseqs_app, sequences)
    cluster_db = mmseqs_app.cluster(database, min_seq_id=0.5, c=0.8, threads=1).result()
    clusters = get_clusters(mmseqs_app, cluster_db)

    # Every sequence appears in exactly one cluster
    members = [member for cluster in clusters.values() for member in cluster]
    assert sorted(members) == sorted(sequences)
    # Each cluster comprises exactly one family
    assert {frozenset(cluster) for cluster in clusters.values()} == {
        frozenset(family) for family in families.values()
    }
    # The representative is a member of its own cluster
    for representative, cluster in clusters.items():
        assert representative in cluster


@pytest.mark.parametrize("n_msas", [1, 3])
def test_msa_roundtrip(mmseqs_app, n_msas):
    """
    MSAs survive a roundtrip through an MSA database unchanged.
    """
    SEQUENCE_LENGTH = 80
    MUTATION_RATE = 0.1

    rng = np.random.default_rng(0)
    msa = {}
    for i in range(n_msas):
        query = seq.ProteinSequence(_random_protein_sequence(rng, SEQUENCE_LENGTH))
        msa[f"msa_{i}"] = _random_msa(rng, query, 4, MUTATION_RATE)

    restored = get_msa_from_database(
        mmseqs_app, create_database_from_msa(mmseqs_app, msa)
    )

    assert sorted(restored) == sorted(msa)
    for identifier, alignments in msa.items():
        assert len(restored[identifier]) == len(alignments)
        for reference, test in zip(alignments, restored[identifier], strict=True):
            assert test.get_gapped_sequences() == reference.get_gapped_sequences()


@pytest.mark.parametrize("seed", range(3))
def test_matrix_from_profile(mmseqs_app, seed):
    """
    The exported position-specific scoring matrix agrees with the log-odds
    matrix of the corresponding Biotite profile.
    As *MMseqs2* applies sequence weighting and pseudo counts, only the ranking
    of the scores is comparable, not their magnitude.
    """
    SEQUENCE_LENGTH = 100
    MUTATION_RATE = 0.1
    N_HOMOLOGS = 20

    rng = np.random.default_rng(seed)
    query = seq.ProteinSequence(_random_protein_sequence(rng, SEQUENCE_LENGTH))
    msa = _random_msa(rng, query, N_HOMOLOGS, MUTATION_RATE)

    profile_db = mmseqs_app.msa_to_profile(
        create_database_from_msa(mmseqs_app, {"query": msa}), threads=1
    ).result()
    matrices = get_matrix_from_profile(mmseqs_app, profile_db)
    assert len(matrices) == 1
    matrix = matrices[0]
    alphabet = seq.ProteinSequence.alphabet
    assert matrix.shape == (SEQUENCE_LENGTH, len(alphabet))
    assert matrix.get_alphabet2() == alphabet
    scores = matrix.score_matrix()
    # Only the canonical amino acids are scored by MMseqs2
    canonical = np.array([alphabet.encode(symbol) for symbol in PROTEIN_ALPHABET])
    ambiguous = np.setdiff1d(np.arange(len(alphabet)), canonical)
    assert not scores[:, ambiguous].any()

    # The same profile as Biotite object
    all_sequences = [query] + [alignment.sequences[1] for alignment in msa]
    counts = np.zeros((SEQUENCE_LENGTH, len(alphabet)), dtype=int)
    for sequence in all_sequences:
        counts[np.arange(SEQUENCE_LENGTH), sequence.code] += 1
    profile = seq.SequenceProfile(
        counts, np.zeros(SEQUENCE_LENGTH, dtype=int), alphabet
    )
    log_odds = profile.log_odds_matrix(pseudocount=1)

    # The highest scoring symbol is the most frequent one at almost every position
    assert (
        np.mean(
            scores[:, canonical].argmax(axis=-1) == counts[:, canonical].argmax(axis=-1)
        )
        > 0.9
    )
    correlation = np.corrcoef(
        scores[:, canonical].ravel(), log_odds[:, canonical].ravel()
    )[0, 1]
    assert correlation > 0.5


@pytest.mark.parametrize("as_3di", [False, True])
def test_database_from_structures(foldseek_app, structure_path, as_3di):
    """
    A database created from `AtomArray` objects gives the same sequences as
    Biotite extracts from them directly.
    """
    atoms = pdbx.get_structure(pdbx.CIFFile.read(structure_path), model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    if as_3di:
        reference_sequences = strucalph.to_3di(atoms)
    else:
        reference_sequences = struc.to_sequence(atoms)
    reference = {
        f"structure_{atoms.chain_id[start]}": str(sequence).upper()
        for sequence, start in zip(*reference_sequences)
    }

    database = create_database_from_structures(
        foldseek_app, {"structure": atoms}, threads=1
    )

    assert get_sequences_from_database(foldseek_app, database, as_3di=as_3di) == (
        reference
    )


def test_multimer_report(foldseek_app, multimer_path):
    """
    A complex searched against itself is matched chain by chain with a perfect
    TM-score.
    """
    atoms = pdbx.get_structure(pdbx.CIFFile.read(multimer_path), model=1)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    chains = sorted(set(atoms.chain_id))
    assert len(chains) > 1

    database = create_database_from_structures(
        foldseek_app, {"complex": atoms}, threads=1
    )
    alignment_db = foldseek_app.search_multimers(database, database, threads=1).result()

    report = get_multimer_report(foldseek_app, alignment_db)

    assert set(report["query"]) == {"complex"}
    assert set(report["target"]) == {"complex"}
    # The first hit is the complex matched to itself, chain by chain
    assert report["query_chains"][0] == ",".join(chains)
    assert report["target_chains"][0] == ",".join(chains)
    assert float(report["query_tm_score"][0]) == pytest.approx(1.0)
    assert float(report["target_tm_score"][0]) == pytest.approx(1.0)
