from os.path import isfile
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
    uids, biotite.temp_file("fasta"), db_name="protein", ret_type="fasta"
)
fasta_file = fasta.FastaFile()
fasta_file.read(file_name)
hit_seqs = [s[:40] for s in fasta.get_sequences().values()]
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


# Plot MSA
number_functions = []
for start in hit_starts:
    def some_func(x, start=start):
        return x + start
    number_functions.append(some_func)
fig = plt.figure(figsize=(8.0, 8.0))
ax = fig.add_subplot(111)
graphics.plot_alignment_type_based(
    ax, alignment, symbols_per_line=len(alignment), labels=hit_ids,
    symbol_size=8, number_size=8, label_size=8,
    show_numbers=True, number_functions=number_functions,
    color_scheme="flower"
)
fig.tight_layout()

# Calculate MI
#def mutual_information(alignment, ref_index):
#    ref_seq = alignment.sequences[ref_index]
#    alph_len = len(ref_seq.alphabet)
#    ref_trace = alignment.trace[:,ref_index]
#    codes = align.get_codes(alignment).T
#    codes = codes[ref_trace != -1]
#    
#    counts = np.zeros((len(codes), alph_len))
#    lengths = np.zeros(len(codes))
#    for i, column in enumerate(codes):
#        counts[i] = np.bincount(column[column != -1], minlength=alph_len)
#        lengths[i] = len(column[column != -1])
#    
#    marginal_probs = counts / lengths[:, np.newaxis]
#    combined_probs = np.zeros((len(ref_seq), len(ref_seq)))
#    
#    mi = np.zeros((len(ref_seq), len(ref_seq)))
#    for i, col_i in enumerate(codes):
#        if ref_trace[i] == -1:
#            continue
#        for j, col_j in enumerate(codes):
#            if ref_trace[j] == -1:
#                continue
#            for k in codes.shape[-1:]


mi = mutual_information(alignment, 0)

plt.show()