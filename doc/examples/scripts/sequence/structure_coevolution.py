import numpy as np
import matplotlib.pyplot as plt
import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import biotite.sequence as seq
import biotite.sequence.io.fasta as fasta
import biotite.sequence.align as align
import biotite.sequence.graphics as graphics
import biotite.application.blast as blast
import biotite.application.clustalo as clustalo
import biotite.database.rcsb as rcsb
import biotite.database.entrez as entrez


IDENTITY_THESHOLD = 0.4

# Get structure and sequence
pdbx_file = pdbx.PDBxFile()
pdbx_file.read(rcsb.fetch("1GUU", "mmcif", "."))
#file.read(rcsb.fetch("1GUU", "mmcif"))
cmyb_seq = pdbx.get_sequence(pdbx_file)[0]
cmyb_struc = pdbx.get_structure(pdbx_file, model=1)

"""
# Find homologous proteins in SwissProt via BLAST
app = blast.BlastWebApp("blastp", cmyb_seq, database="swissprot")
app.start()
app.join()
alignments = app.get_alignments()
hit_seqs = [cmyb_seq]
hit_ids = ["Query"]
hit_starts = [1]
for ali in alignments:
    identity = align.get_sequence_identity(ali)
    if identity > IDENTITY_THESHOLD and identity < 1.0:
        hit_seqs.append(ali.sequences[1])
        hit_ids.append(ali.hit_id)
        hit_starts.append(ali.hit_interval[0])
"""
###
# Search for protein products of LexA gene in UniProtKB/Swiss-Prot database
query =   entrez.SimpleQuery("luxA", "Gene Name") \
        & entrez.SimpleQuery("srcdb_swiss-prot", "Properties")
uids = entrez.search(query, db_name="protein")
file_name = entrez.fetch_single_file(
    uids, "test.fasta", db_name="protein", ret_type="fasta"
)
fasta_file = fasta.FastaFile()
fasta_file.read(file_name)
hit_seqs = [s[:40] for s in fasta.get_sequences(fasta_file).values()]
hit_ids = ["TEST"] * len(hit_seqs)
hit_starts = [1] * len(hit_seqs)
###

ids = []
sequences = []
for header, seq_str in fasta_file.items():
    # Extract the UniProt Entry name from header
    identifier = header.split("|")[-1].split()[0]
    ids.append(identifier)
    sequences.append(seq.ProteinSequence(seq_str))

# Perform MSA
alignment = clustalo.ClustalOmegaApp.align(hit_seqs)
###
#alignment = alignment[alignment.trace[:,0] != -1]
#alignment = alignment[:1]
###


# Plot MSA
number_functions = []
for start in hit_starts:
    def some_func(x, start=start):
        return x + start
    number_functions.append(some_func)
fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.gca()
graphics.plot_alignment_type_based(
    ax, alignment, symbols_per_line=len(alignment), labels=hit_ids,
    symbol_size=8, number_size=8, label_size=8,
    show_numbers=True, number_functions=number_functions,
    color_scheme="flower"
)
fig.tight_layout()

# Calculate MI
def mutual_information(alignment):
    codes = align.get_codes(alignment).T
    alph = alignment.sequences[0].alphabet
    
    mi = np.zeros((len(alignment), len(alignment)))
    # Iterate over all columns to choose first column
    for i in range(codes.shape[0]):
        # Iterate over all columns to choose second column
        for j in range(codes.shape[0]):
            nrows = 0
            marginal_counts_i = np.zeros(len(alph), dtype=int)
            marginal_counts_j = np.zeros(len(alph), dtype=int)
            combined_counts = np.zeros((len(alph), len(alph)), dtype=int)
            # Iterate over all symbols in both columns
            for k in range(codes.shape[1]):
                if codes[i,k] != -1 and codes[j,k] != -1:
                    marginal_counts_i[codes[i,k]] += 1
                    marginal_counts_j[codes[j,k]] += 1
                    combined_counts[codes[i,k], codes[j,k]] += 1
                    nrows += 1
            marginal_probs_i = marginal_counts_i / nrows
            marginal_probs_j = marginal_counts_j / nrows
            combined_probs = combined_counts / nrows
            
            mi_before_sum = (
                combined_probs * np.log2(
                    combined_probs / (
                        marginal_probs_i[:, np.newaxis] * 
                        marginal_probs_j[np.newaxis, :]
                    )
                )
            ).flatten()
            mi[i,j] = np.sum(mi_before_sum[~np.isnan(mi_before_sum)])
    return mi

alignment = alignment[alignment.trace[:,0] != -1]
mi = mutual_information(alignment)

fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.gca()
#cmap = ListedColormap(["white", biotite.colors["dimgreen"]])
cmap="Greens"
#ax.matshow(adjacency_matrix, cmap=cmap, origin="lower")
im = ax.pcolormesh(mi, cmap=cmap)
ax.set_aspect("equal")
ax.set_xlabel("Residue position")
ax.set_xlabel("Residue position")
ax.set_title("Mutual information")
fig.colorbar(im)
fig.tight_layout()


plt.show()