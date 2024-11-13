"""
Note that in order to get a representative sequence profile, it is crucial to have a
high number of sequences the target sequence can be aligned to.
"""

import matplotlib.pyplot as plt
import numpy as np
import requests
import biotite.database.uniprot as uniprot
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.sequence.io.fasta as fasta
from biotite.database import RequestError

TARGET_UNIPROT_ID = "P04746"
TARGET_SCOP_FAMILY_ID = "4003138"
QUERY_UNIPROT_ID = "P00722"

# TARGET_UNIPROT_ID = "P09467"
# TARGET_SCOP_FAMILY_ID = "4002766"
# QUERY_UNIPROT_ID = "P73922"


target_sequence = fasta.get_sequence(
    fasta.FastaFile.read(uniprot.fetch(TARGET_UNIPROT_ID, "fasta", "."))
)
query_sequence = fasta.get_sequence(
    fasta.FastaFile.read(uniprot.fetch(QUERY_UNIPROT_ID, "fasta", "."))
)

aa_matrix = align.SubstitutionMatrix.std_protein_matrix()
alignment = align.align_optimal(
    query_sequence,
    target_sequence,
    aa_matrix,
    gap_penalty=(-10, -1),
    local=False,
    max_number=1,
)[0]

fig, ax = plt.subplots(figsize=(8.0, 1.0))
graphics.plot_alignment_similarity_based(
    ax, alignment, matrix=aa_matrix, labels=[TARGET_UNIPROT_ID, QUERY_UNIPROT_ID]
)


########################################################################################
# https://www.ebi.ac.uk/pdbe/scop/download


def get_sequence_family(scop_id):
    scop_domains = requests.get(
        f"https://www.ebi.ac.uk/pdbe/scop/api/domains/{scop_id}"
    ).json()
    sequences = {}
    for scop_domain in scop_domains["domains"]:
        uniprot_id = scop_domain["uniprot_id"]
        try:
            sequence = fasta.get_sequence(
                fasta.FastaFile.read(uniprot.fetch(uniprot_id, "fasta", "."))
            )
        except RequestError:
            # The UniProt ID is not in the database -> skip this domain
            continue
        for start, end in scop_domain["protein_regions"]:
            region = sequence[start : end + 1]
            sequences[uniprot_id] = region
            # Most domains have only one region within the sequence
            # For simplicity we only consider the first region
            break
    return sequences


def merge_pairwise_alignments(alignments):
    traces = []
    sequences = []
    for alignment in alignments:
        trace = alignment.trace
        # Remove gaps in first sequence
        trace = trace[trace[:, 0] != -1]
        traces.append(trace[:, 1])
        sequences.append(alignment.sequences[1])
    return align.Alignment(sequences, np.stack(traces, axis=-1))


sequences = get_sequence_family(TARGET_SCOP_FAMILY_ID)
# This is not a 'true' MSA, in the sense that it only merges the pairwise alignments
# with respect to the target sequence
pseudo_msa = merge_pairwise_alignments(
    [
        align.align_optimal(
            target_sequence,
            sequence,
            aa_matrix,
            gap_penalty=(-10, -1),
            max_number=1,
        )[0]
        for sequence in sequences.values()
    ]
)

labels = np.array(list(sequences.keys()))

fig, ax = plt.subplots(figsize=(8.0, 24.0))
graphics.plot_alignment_type_based(ax, pseudo_msa, labels=labels, show_numbers=True)

########################################################################################
# :footcite:`Robinson1991`

# Must have the same order as `ProteinSequence.alphabet`
AA_FREQUENCY = {
    "A": 35155,
    "C": 8669,
    "D": 24161,
    "E": 28354,
    "F": 17367,
    "G": 33229,
    "H": 9906,
    "I": 23161,
    "K": 25872,
    "L": 40625,
    "M": 10101,
    "N": 20212,
    "P": 23435,
    "Q": 19208,
    "R": 23105,
    "S": 32070,
    "T": 26311,
    "V": 29012,
    "W": 5990,
    "Y": 14488,
    # Set ambiguous amino acid count to 1 to avoid division by zero in log odds matrix
    "B": 1,
    "Z": 1,
    "X": 1,
    "*": 1,
}


profile = seq.SequenceProfile.from_alignment(pseudo_msa)
background_frequencies = np.array(list(AA_FREQUENCY.values()))
# Normalize background frequencies
background_frequencies = background_frequencies / background_frequencies.sum()

score_matrix = profile.log_odds_matrix(background_frequencies, pseudocount=1).T
score_matrix *= 10
score_matrix = score_matrix.astype(np.int32)

profile_seq = seq.PositionalSequence(profile.to_consensus())
positional_matrix = align.SubstitutionMatrix(
    seq.ProteinSequence.alphabet, profile_seq.alphabet, score_matrix=score_matrix
)
profile_alignment = align.align_optimal(
    query_sequence,
    profile_seq,
    positional_matrix,
    gap_penalty=(-10, -1),
    local=False,
    terminal_penalty=False,
    max_number=1,
)[0]


########################################################################################
# Map the profile alignment to the original target sequence.
# As the pseudo-MSA was designed to have the same length as the target sequence,
# the sequence needs to be exchanged only, the trace remains the same
refined_alignment = align.Alignment(
    [query_sequence, target_sequence], profile_alignment.trace
)

fig, ax = plt.subplots(figsize=(8.0, 4.0))
graphics.plot_alignment_similarity_based(
    ax,
    refined_alignment,
    matrix=aa_matrix,
    labels=[TARGET_UNIPROT_ID, QUERY_UNIPROT_ID],
)


plt.show()

########################################################################################
#
# .. footbibliography::
