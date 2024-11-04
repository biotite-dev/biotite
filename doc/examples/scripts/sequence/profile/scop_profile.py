import matplotlib.pyplot as plt
import numpy as np
import requests
import biotite.application.clustalo as clustalo
import biotite.database.uniprot as uniprot
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.sequence.io.fasta as fasta
from biotite.database import RequestError

TARGET_UNIPROT_ID = "P10145"
TARGET_SCOP_FAMILY_ID = "4000912"
QUERY_UNIPROT_ID = "P03212"

#TARGET_UNIPROT_ID = "P09467"
#TARGET_SCOP_FAMILY_ID = "4002766"
#QUERY_UNIPROT_ID = "P73922"


target_sequence = fasta.get_sequence(
    fasta.FastaFile.read(uniprot.fetch(TARGET_UNIPROT_ID, "fasta", "."))
)
query_sequence = fasta.get_sequence(
    fasta.FastaFile.read(uniprot.fetch(QUERY_UNIPROT_ID, "fasta", "."))
)

aa_matrix = align.SubstitutionMatrix.std_protein_matrix()
alignment = align.align_optimal(
    target_sequence,
    query_sequence,
    aa_matrix,
    gap_penalty=(-10, -1),
    local=True,
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


sequences = get_sequence_family(TARGET_SCOP_FAMILY_ID)
app = clustalo.ClustalOmegaApp(list(sequences.values()))
app.start()
app.join()
msa = app.get_alignment()
order = app.get_alignment_order()
labels = np.array(list(sequences.keys()))

fig, ax = plt.subplots(figsize=(8.0, 24.0))
graphics.plot_alignment_type_based(
    ax, msa[:, order], labels=labels[order], show_numbers=True
)

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


profile = seq.SequenceProfile.from_alignment(msa)
background_frequencies = np.array(list(AA_FREQUENCY.values()))
# Normalize background frequencies
background_frequencies = background_frequencies / background_frequencies.sum()

score_matrix = profile.log_odds_matrix(background_frequencies, pseudocount=1)
score_matrix *= 10
score_matrix = score_matrix.astype(np.int32)

profile_seq = seq.PositionalSequence(profile.to_consensus())
positional_matrix = align.SubstitutionMatrix(
    profile_seq.alphabet, seq.ProteinSequence.alphabet, score_matrix=score_matrix
)
profile_alignment = align.align_optimal(
    profile_seq,
    query_sequence,
    positional_matrix,
    gap_penalty=(-10, -1),
    local=False,
    terminal_penalty=False,
    max_number=1,
)[0]
print(profile_alignment)


########################################################################################
# Map the profile alignment to the original target sequence
# Chimeric alignment from target in the MSA and the query in the profile alignment
sequence_pos_in_msa = np.where(labels == TARGET_UNIPROT_ID)[0][0]
target_trace = msa.trace[:, sequence_pos_in_msa]
# Remove parts of query aligned to gaps in in the profile
gap_mask = profile_alignment.trace[:, 0] != -1
query_trace = profile_alignment.trace[gap_mask, 1]
print(msa.trace.shape)
print(profile_alignment.trace.shape)
print(profile.symbols.shape)
print()
print(query_trace.shape)
print(target_trace.shape)
target_alignment = align.Alignment(
    [target_sequence, query_sequence],
    np.stack([target_trace, query_trace], axis=-1),
)

fig, ax = plt.subplots(figsize=(8.0, 4.0))
graphics.plot_alignment_similarity_based(
    ax, target_alignment, matrix=aa_matrix, labels=[TARGET_UNIPROT_ID, QUERY_UNIPROT_ID]
)


plt.show()

########################################################################################
#
# .. footbibliography::
