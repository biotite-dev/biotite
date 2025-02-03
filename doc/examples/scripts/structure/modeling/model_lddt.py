r"""
LDDT for predicted structure evaluation
=======================================

This example evaluates the quality of a predicted structure from *AlphaFold DB* compared
to the experimental structure of a protein of interest by the means of the lDDT score.
Furthermore, the measured lDDT score is compared to the pLDDT score predicted by the
model.
"""

# Code source: Patrick Kunzmann
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import biotite
import biotite.database.afdb as afdb
import biotite.database.rcsb as rcsb
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx

# Uniprot ID of the protein of interest (in this case human beta-galactosidase)
UNIPROT_ID = "P16278"


## Get the reference experimental structure from the PDB
query = rcsb.FieldQuery(
    "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
    exact_match=UNIPROT_ID,
)
# The UniProt ID is defined for a single chain
ids = rcsb.search(query, return_type="polymer_instance")
# Simply use the first matching chain as reference
pdb_id, chain_id = ids[0].split(".")
pdbx_file = pdbx.BinaryCIFFile.read(rcsb.fetch(pdb_id, "bcif"))
reference = pdbx.get_structure(pdbx_file, model=1, use_author_fields=False)
reference = reference[reference.chain_id == chain_id]
# The experimental structure may contain additional small molecules
# (e.g. water, ions etc.) that are not part of the predicted structure
reference = reference[struc.filter_amino_acids(reference)]


## Get the predicted structure from AlphaFold DB
pdbx_file = pdbx.BinaryCIFFile.read(afdb.fetch(UNIPROT_ID, "bcif"))
# Use 'label_<x>' fields to make sure the residue ID is the the same as given in the
# `ma_qa_metric_local` category, where the pLDDT is obtained from
model = pdbx.get_structure(pdbx_file, model=1, use_author_fields=False)


## Filter the structures to common atoms that are present in both structures
reference_sequence = struc.to_sequence(reference)[0][0]
model_sequence = struc.to_sequence(model)[0][0]
# This script does not rely on consistent residue numbering,
# so a sequence alignment is done instead
identity_matrix = align.SubstitutionMatrix(
    seq.ProteinSequence.alphabet,
    seq.ProteinSequence.alphabet,
    np.eye(len(seq.ProteinSequence.alphabet), dtype=int),
)
alignment = align.align_optimal(
    reference_sequence,
    model_sequence,
    # Residues might be missing due to experimental reasons but not due to homology
    # -> use a simple identity matrix
    identity_matrix,
    gap_penalty=-1,
    terminal_penalty=False,
    max_number=1,
)[0]
# Remove residues from alignment
# that have no correspondence in the respective other structure
# -> Remove gaps (-1 entries in trace)
alignment = alignment[(alignment.trace != -1).all(axis=1)]
# Map the remaining alignment columns to atom indices
reference = reference[
    # Each mask is True for all atoms in one residue
    struc.get_residue_masks(reference, struc.get_residue_starts(reference)) \
    # Only keep masks for residues that correspond to remaining alignment columns
    [alignment.trace[:,0]] \
    # And aggregate them to get a single mask
    .any(axis=0)
]  # fmt: skip
model = model[
    struc.get_residue_masks(model, struc.get_residue_starts(model))[
        alignment.trace[:, 1]
    ].any(axis=0)
]


## Get predicted lDDT from the model file
plddt_category = pdbx_file.block["ma_qa_metric_local"]
plddt_res_ids = plddt_category["label_seq_id"].as_array(int)
plddt = plddt_category["metric_value"].as_array(float) / 100
# Remove values for residues that were removed in the alignment process
mask = np.isin(plddt_res_ids, model.res_id)
plddt_res_ids = plddt_res_ids[mask]
plddt = plddt[mask]


## Compute actual lDDT by comparing the model to the reference
lddt_res_ids = np.unique(model.res_id)
# The pLDDT predicts the lDDT of CA atoms, so for consistency we do the same
ca_mask = model.atom_name == "CA"
lddt = struc.lddt(reference[ca_mask], model[ca_mask], aggregation="residue")


## Compare predicted to measured lDDT
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.plot(
    plddt_res_ids,
    plddt,
    color=biotite.colors["dimgreen"],
    linestyle="-",
    label="predicted",
)
ax.plot(
    lddt_res_ids,
    lddt,
    color=biotite.colors["lightorange"],
    linestyle="-",
    label="measured",
)
ax.legend()
ax.set_xlabel("Residue ID")
ax.set_ylabel("lDDT")
ax.autoscale(axis="x", tight=True)
plt.show()
