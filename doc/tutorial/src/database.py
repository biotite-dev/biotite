"""
Data to work with - The Database subpackage
===========================================

.. currentmodule:: biotite

Biological databases are the backbone of computational biology.
The :mod:`biotite.database` subpackage provides interfaces for popular
online databases like the *RCSB PDB* or the *NCBI Entrez* database.

Fetching structure files from the RCSB PDB
------------------------------------------

.. currentmodule:: biotite.database.rcsb

Downloading structure files from the *RCSB PDB* is quite easy:
Simply specify the PDB ID, the file format and the target directory
for the :func:`fetch()` function and you are done.
The function returns the path to the downloaded file, so you
can simply load the file via the other *Biotite* subpackages
(more on this later).
We will download on a protein structure of the miniprotein *TC5b*
(PDB: 1L2Y) into a temporary directory.
"""

from tempfile import gettempdir
import biotite.database.rcsb as rcsb

file_path = rcsb.fetch("1l2y", "pdb", gettempdir())
print(file_path)

########################################################################
# In case you want to download multiple files, you are able to specify a
# list of PDB IDs, which in return gives you a list of file paths.

# Download files in the more modern mmCIF format
file_paths = rcsb.fetch(["1l2y", "1aki"], "cif", gettempdir())
print([file_path for file_path in file_paths])

########################################################################
# By default :func:`fetch()` checks whether the file to be fetched
# already exists in the directory and downloads it, if it does not
# exist yet.
# If you want to download files irrespectively, set :obj:`overwrite` to
# true.

# Download file in the fast and small binary MMTF format
file_path = rcsb.fetch("1l2y", "mmtf", gettempdir(), overwrite=True)

########################################################################
# If you omit the file path or set it to ``None``, the downloaded data
# will be returned directly as a file-like object, without creating a
# file on your disk at all.

file = rcsb.fetch("1l2y", "pdb")
lines = file.readlines()
print("\n".join(lines[:10] + ["..."]))

########################################################################
# In many cases you are not interested in a specific structure, but you
# want a set of structures that fits your desired criteria.
# For this purpose the *RCSB* search API can be used.
# At first you have to create :class:`Query` object for the property you
# want to filter.
# The :func:`search()` method takes the :class:`Query` and returns a
# list of PDB IDs, which itself can be used as input for
# :func:`fetch()`.
# Likewise, :func:`count()` is used to count the number of matching
# PDB IDs.

query = rcsb.BasicQuery("HCN1")
pdb_ids = rcsb.search(query)
print(pdb_ids)
print(rcsb.count(query))
files = rcsb.fetch(pdb_ids, "mmtf", gettempdir())

########################################################################
# This was a simple search for the occurrence of the search term in any
# field. 
# You can also search for a value in a specific field with a
# :class:`FieldQuery`.
# A complete list of the available fields and its supported operators
# is documented
# `on this page <https://search.rcsb.org/structure-search-attributes.html>`_
# and `on that page <https://search.rcsb.org/chemical-search-attributes.html>`.

# Query for 'lacA' gene
query1 = rcsb.FieldQuery(
    "rcsb_entity_source_organism.rcsb_gene_name.value",
    exact_match="lacA"
)
# Query for resolution below 1.5 Å
query2 = rcsb.FieldQuery("reflns.d_resolution_high", less=1.5)

########################################################################
# The search API allows even more complex queries, e.g. for sequence
# or structure similarity. Have a look at the API reference of
# :mod:`biotite.database.rcsb`.
#
# Multiple :class:`Query` objects can be combined using the ``|`` (or)
# or ``&`` (and) operator for a more fine-grained selection.
# A :class:`FieldQuery` is negated with ``~``.

composite_query = query1 & ~query2
print(rcsb.search(composite_query))

########################################################################
# Often the structures behind the obtained PDB IDs have degree of
# redundancy.
# For example they may represent the same protein sequences or result
# from the same set of experiments.
# You may use :class:`Grouping` of structures to group redundant
# entries or even return only single representatives of each group.

query = rcsb.BasicQuery("Transketolase")
# Group PDB IDs from the same collection
print(rcsb.search(
    query, group_by=rcsb.DepositGrouping(), return_groups=True
))
# Get only a single representative of each group
print(rcsb.search(
    query, group_by=rcsb.DepositGrouping(), return_groups=False
))

########################################################################
# Note that grouping may omit PDB IDs in search results, if such PDB IDs
# cannot be grouped.
# In the example shown above, not all structures 
# For example in the case shown above only a few PDB entries were
# uploaded as collection and hence are part of the search results.
#
# Fetching files from the NCBI Entrez database
# --------------------------------------------
# 
# .. currentmodule:: biotite.database.entrez
# 
# Another important source of biological information is the
# *NCBI Entrez* database, which is commonly known as *the NCBI*.
# It provides a myriad of information, ranging from sequences and
# sequence features to scientific articles.
# Fetching files from NCBI Entrez works analogous to the RCSB interface.
# This time we have to provide the UIDs (Accession or GI) instead of
# PDB IDs to the :func:`fetch()` function.
# Furthermore, we need to specifiy the database to retrieve the data
# from and the retrieval type.

from tempfile import gettempdir, NamedTemporaryFile
import biotite.database.entrez as entrez

# Fetch a single UID ...
file_path = entrez.fetch(
    "NC_001416", gettempdir(), suffix="fa",
    db_name="nuccore", ret_type="fasta"
)
print(file_path)
# ... or multiple UIDs 
file_paths = entrez.fetch(
    ["1L2Y_A","1AKI_A"], gettempdir(), suffix="fa",
    db_name="protein", ret_type="fasta"
)
print([file_path for file_path in file_paths])

########################################################################
# A list of valid database, retrieval type and mode combinations can
# be found
# `here <https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly>`_.
# Furthermore, :func:`get_database_name()` can be helpful to get the
# required database name by the more commonly known names.

print(entrez.get_database_name("Nucleotide"))

########################################################################
# The *Entrez* database allows for packing data for multiple UIDs into a
# single file. This is achieved with the :func:`fetch_single_file()`
# function.

temp_file = NamedTemporaryFile(suffix=".fasta", delete=False)
file_path = entrez.fetch_single_file(
    ["1L2Y_A","1AKI_A"], temp_file.name, db_name="protein", ret_type="fasta"
)
print(file_path)
temp_file.close()

########################################################################
# Similar to the *RCSB PDB*, you can also search every
# `field <https://www.ncbi.nlm.nih.gov/books/NBK49540/>`_
# of the *NCBI Entrez* database.

# Search in all fields
print(entrez.SimpleQuery("BL21 genome"))
# Search in the 'Organism' field
print(entrez.SimpleQuery("Escherichia coli", field="Organism"))

########################################################################
# You can also combine multiple :class:`Query` objects in any way you
# like using the binary operators ``|``, ``&`` and ``^``,
# that represent ``OR``,  ``AND`` and ``NOT`` linkage, respectively.

composite_query = (
    entrez.SimpleQuery("50:100", field="Sequence Length") &
    (
        entrez.SimpleQuery("Escherichia coli", field="Organism") |
        entrez.SimpleQuery("Bacillus subtilis", field="Organism")
    )
)
print(composite_query)


########################################################################
# Finally, the query is given to the :func:`search()` function to obtain
# the GIs, that can be used as input for :func:`fetch()`.

# Return a maximum number of 10 entries
gis = entrez.search(composite_query, "protein", number=10)
print(gis)