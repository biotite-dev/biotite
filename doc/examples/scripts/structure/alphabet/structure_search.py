"""
Searching for structural homologs in a protein structure database
=================================================================

The following script implements a protein structure search algorithm akin to `foldseek`
:footcite:`VanKempen2024`.
It harnesses the *3Di* alphabet to translate 3D structures into a sequences.
Structurally homologous regions between these *3Di* sequences can be identified quickly
using *k-mer* matching followed by ungapped seed extension and finally gapped alignment.

While the following example could also be scaled up easily to a large structure
database, for the sake of simplicity we will use a random selection of structures from
the PDB plus an expected target structure, which is known to be structurally homologous
to the query structure.
The aim of the script is to identify this target among the decoy structures in the
dataset.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import collections
import matplotlib.pyplot as plt
import numpy as np
import biotite
import biotite.database.rcsb as rcsb
import biotite.interface.pymol as pymol_interface
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.structure as struc
import biotite.structure.alphabet as strucalph
import biotite.structure.io.pdbx as pdbx

# Amino acid background frequencies
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

# Two protein structures from the same SCOP superfamily but different families
QUERY_ID = "4R8K"  # Asparaginase (Cavia porcellus)
QUERY_CHAIN = "B"
TARGET_ID = "1NNS"  # L-asparaginase 2 (Escherichia coli)
TARGET_CHAIN = "A"
N_DECOYS = 50
# Approximate size (number of amino acids) of AlphaFold DB:
# For E-value estimation assume that the hypothetical database has this size
DB_SIZE = 2e13

## Alignment search parameters
K = 6  # adopted from `foldseek`
SPACED_SEED = "11*1*1**11"  # adopted from `foldseek`
# The number of consecutive undefined 3Di symbols ('d'),
# that will be marked as undefined spans
UNDEFINED_SPAN_LENGTH = 4
X_DROP = 20
X_DROP_ACCEPT_THRESHOLD = 100
BAND_WIDTH = 20
GAP_PENALTY = (-10, -1)
EVALUE_THRESHOLD = 1e-1

########################################################################################
# Limitation of protein sequence alignment
# ----------------------------------------
#
# To motivate using a structural alphabet, we first try a regular sequence alignment
# on the two homologs.


def get_protein_chain(pdb_id, chain_id):
    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))
    # Bonds are required for the later molecular visualization
    atoms = pdbx.get_structure(pdbx_file, model=1, include_bonds=True)
    atoms = atoms[struc.filter_amino_acids(atoms)]
    return atoms[atoms.chain_id == chain_id]


query_aa_sequence = struc.to_sequence(get_protein_chain(QUERY_ID, QUERY_CHAIN))[0][0]
target_aa_sequence = struc.to_sequence(get_protein_chain(TARGET_ID, TARGET_CHAIN))[0][0]

alignment = align.align_optimal(
    query_aa_sequence,
    target_aa_sequence,
    align.SubstitutionMatrix.std_protein_matrix(),
    gap_penalty=(-10, -1),
    local=True,
)[0]

print(f"Sequence identity: {int(100 * align.get_sequence_identity(alignment))}%")
fig, ax = plt.subplots(figsize=(8.0, 3.0), constrained_layout=True)
graphics.plot_alignment_similarity_based(
    ax,
    alignment,
    matrix=align.SubstitutionMatrix.std_protein_matrix(),
    labels=["Query", "Target"],
    show_numbers=True,
    show_line_position=True,
)

########################################################################################
# The sequence identity is quite low.
# One would typically refer to this range as the *twilight zone*, where it is unclear if
# the two sequences are actually homologs.
# If the database comprised the entire AlphaFold DB :footcite:`Varadi2024`, the
# probability of finding an alignment as good as this one by chance would be quite
# high as the E-value suggests below.
#
# The background frequencies for E-value calculation are taken from
# :footcite:`Robinson1991`.

background = np.array(list(AA_FREQUENCY.values()))
# Normalize background frequencies
background = background / background.sum()
np.random.seed(0)
estimator = align.EValueEstimator.from_samples(
    seq.ProteinSequence.alphabet,
    align.SubstitutionMatrix.std_protein_matrix(),
    GAP_PENALTY,
    background,
    sample_size=500,
)
log_evalue = estimator.log_evalue(alignment.score, len(query_aa_sequence), DB_SIZE)
print(f"E-value: {10**log_evalue:.2e}")

########################################################################################
# Sequence preparation
# --------------------
#
# Now the actual task:
# For simplicity we compile a small database of random structures from the PDB,
# that represents the database we want to search in.
# Into this database we shuffle the target structure to which we expect a hit for
# our query.
# To get a list of PDB entries we can select from, we search for all entries with at
# least one peptide chain.
# Furthermore, we want the resolution be be sufficiently good to resolve side chains.

pdb_query = rcsb.FieldQuery(
    "rcsb_entry_info.polymer_entity_count_protein", greater_or_equal=1
) & rcsb.FieldQuery("reflns.d_resolution_high", less=2.0)
# For the sake of performance, only the first 10000 matches are requested from the PDB
protein_pdb_ids = rcsb.search(pdb_query, range=(0, 10000))
# From these 10000, randomly select the decoys
rng = np.random.default_rng(0)
decoy_pdb_ids = rng.choice(protein_pdb_ids, size=100, replace=False)
# Mix the target structure to the decoy set
database_pdb_ids = np.concatenate((decoy_pdb_ids, [TARGET_ID]))
rng.shuffle(database_pdb_ids)

########################################################################################
# Now the selected PDB entries are downloaded and for each peptide chain the *3Di*
# sequence that reflects the structure is determined.
# We will identify each sequence by a combination of the PDB ID and chain ID.

db_ids = []
db_sequences = []
for pdb_id in database_pdb_ids:
    pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))
    # Use only one model per structure,
    # as most models will probably be very similar to each other anyway
    # It is also important to use the 'auth_asym_id' for consistency with
    # 'entity_poly.pdbx_strand_id' below
    atoms = pdbx.get_structure(pdbx_file, model=1, use_author_fields=True)
    entity_poly = pdbx_file.block["entity_poly"]
    for i in range(entity_poly.row_count):
        chains_in_entity = entity_poly["pdbx_strand_id"].as_array(str)[i]
        # Do not include duplicate sequences, e.g. from two chains of a homodimer
        representative_chain_id = chains_in_entity[0]
        chain = atoms[atoms.chain_id == representative_chain_id]
        # Only use the peptide part of the structure
        chain = chain[struc.filter_amino_acids(chain)]
        if chain.array_length() == 0:
            continue
        # Since we input a single chain, we know only one sequence is created
        structural_sequence = strucalph.to_3di(chain)[0][0]
        if len(structural_sequence) < len(SPACED_SEED):
            # Cannot index a sequence later, if it is shorter than the k-mer span
            continue
        db_ids.append((pdb_id, representative_chain_id))
        db_sequences.append(structural_sequence)

########################################################################################
# Alignment search
# ----------------
#
# The following steps are typical for alignment search methods and are performed
# in a similar fashion as in other applications, such as
# :doc:`classical alignment searches <../../sequence/homology/genome_search>` or
# :doc:`comparative genome assembly <../../sequence/sequencing/genome_assembly>`.
#
# The fast *k-mer* matching requires indexing the *k-mers* in the database.
# Here we want to filter out long spans of undefined symbols, as these often result from
# unresolved residues.
# If the query also had unresolved residues, this would result in spurious matches.


def filter_undefined_spans(sequence, min_length):
    undefined_code = sequence.alphabet.encode(strucalph.I3DSequence.undefined_symbol)
    is_undefined = sequence.code == undefined_code
    return np.all(
        np.stack([np.roll(is_undefined, -offset) for offset in range(min_length)]),
        axis=0,
    )


kmer_table = align.KmerTable.from_sequences(
    K,
    db_sequences,
    spacing=SPACED_SEED,
    ignore_masks=[
        filter_undefined_spans(sequence, UNDEFINED_SPAN_LENGTH)
        for sequence in db_sequences
    ],
)

########################################################################################
# Now we download the query structure and convert it into a *3Di* sequence as well.

pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(QUERY_ID, "bcif"))
atoms = pdbx.get_structure(pdbx_file, model=1)
chain = atoms[atoms.chain_id == QUERY_CHAIN]
chain = chain[struc.filter_amino_acids(chain)]
query_sequence = strucalph.to_3di(chain)[0][0]

########################################################################################
# As a fast first first step we will match the *k-mers* of the query sequence against
# the index of the database.
# To further boil the matches down to the most promising ones, we require double hits
# on the same diagonal.

matches = kmer_table.match(
    query_sequence,
    ignore_mask=filter_undefined_spans(query_sequence, UNDEFINED_SPAN_LENGTH),
)

diagonals = matches[:, 2] - matches[:, 0]
matches_for_diagonals = collections.defaultdict(list)
for i, (diag, db_index) in enumerate(zip(diagonals, matches[:, 1])):
    matches_for_diagonals[(diag, db_index)].append(i)

# If a diagonal has more than one match,
# the first match on this diagonal is a double hit
double_hit_indices = [
    indices[0] for indices in matches_for_diagonals.values() if len(indices) > 1
]
double_hits = matches[double_hit_indices]

########################################################################################
# As second step we perform a quick ungapped alignment of the query sequence against
# the matched database sequences from the previous step.
# We only keep those hits that meet the given score threshold.

substitution_matrix = align.SubstitutionMatrix.std_3di_matrix()
ungapped_scores = np.array(
    [
        align.align_local_ungapped(
            query_sequence,
            db_sequences[db_index],
            substitution_matrix,
            seed=(i, j),
            threshold=X_DROP,
            score_only=True,
        )
        for i, db_index, j in double_hits
    ]
)
accepted_hits = double_hits[ungapped_scores > X_DROP_ACCEPT_THRESHOLD]

########################################################################################
# Finally we perform a gapped alignments at the remaining few hits.
# To avoid computing the complete alignment search space, we use a banded alignment
# with a fixed band width.

# Use the symbol frequencies in the database for E-value estimation
background = np.zeros(len(strucalph.I3DSequence.alphabet), dtype=int)
for sequence in db_sequences:
    background += np.bincount(
        sequence.code, minlength=len(strucalph.I3DSequence.alphabet)
    )
np.random.seed(0)
estimator_for_3di = align.EValueEstimator.from_samples(
    strucalph.I3DSequence.alphabet,
    substitution_matrix,
    GAP_PENALTY,
    background,
    sample_size=500,
)

significant_alignments = {}
for query_pos, db_index, db_pos in accepted_hits:
    diagonal = db_pos - query_pos
    alignment = align.align_banded(
        query_sequence,
        db_sequences[db_index],
        substitution_matrix,
        band=(diagonal - BAND_WIDTH, diagonal + BAND_WIDTH),
        gap_penalty=GAP_PENALTY,
        local=True,
        max_number=1,
    )[0]
    log_evalue = estimator_for_3di.log_evalue(
        alignment.score, len(query_sequence), DB_SIZE
    )
    if log_evalue <= np.log10(EVALUE_THRESHOLD):
        # Keep only one alignment per database sequence
        significant_alignments[db_index] = (alignment, log_evalue)

########################################################################################
# What remains is a single highly significant alignment: the one we have been expecting.
# Its coverage and sequence identity are much higher than in the corresponding protein
# sequence alignment above, which is also reflected in the low E-value.

for db_index, (alignment, log_evalue) in significant_alignments.items():
    print("Aligned target:", db_ids[db_index])
    print(f"E-value: {10**log_evalue:.2e}")
    print(f"Sequence identity: {int(100 * align.get_sequence_identity(alignment))}%")
    fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
    graphics.plot_alignment_similarity_based(
        ax,
        alignment,
        matrix=align.SubstitutionMatrix.std_protein_matrix(),
        labels=["Query", "Target"],
        show_numbers=True,
        show_line_position=True,
        color=biotite.colors["lightorange"],
    )

plt.show()

########################################################################################
# Structure superimposition based on alignment
# --------------------------------------------
#
# As bonus, we can use the aligned residues as *'anchors'* for structure
# superimposition.
# This means, that we superimpose the :math:`C_{\alpha}` atoms of the aligned residues
# and apply the resulting transformation to the full structure.

# In this example we know there is only one significant alignment
# and this corresponds to the query and target structure
expected_alignment, _ = next(iter(significant_alignments.values()))
anchor_indices = expected_alignment.trace
anchor_indices = anchor_indices[(anchor_indices != -1).all(axis=1)]

query_chain = get_protein_chain(QUERY_ID, QUERY_CHAIN)
target_chain = get_protein_chain(TARGET_ID, TARGET_CHAIN)
query_anchors = query_chain[query_chain.atom_name == "CA"][anchor_indices[:, 0]]
target_anchors = target_chain[target_chain.atom_name == "CA"][anchor_indices[:, 1]]
# Find transformation that superimposes the anchor atoms
_, transform = struc.superimpose(query_anchors, target_anchors)
# Apply this transformation to full structure
target_chain = transform.apply(target_chain)

# Visualization with PyMOL
pymol_interface.cmd.set("cartoon_rect_length", 1.0)
pymol_interface.cmd.set("depth_cue", 0)
pymol_interface.cmd.set("cartoon_cylindrical_helices", 1)
pymol_interface.cmd.set("cartoon_helix_radius", 1.5)
pymol_query = pymol_interface.PyMOLObject.from_structure(query_chain)
pymol_target = pymol_interface.PyMOLObject.from_structure(target_chain)
pymol_query.show_as("cartoon")
pymol_target.show_as("cartoon")
pymol_query.color("biotite_lightgreen")
pymol_target.color("biotite_lightorange")
pymol_query.orient()
pymol_interface.show((1500, 1000))
# sphinx_gallery_thumbnail_number = 3

########################################################################################
#
# .. footbibliography::
#
