"""
Data to work with - The Database subpackage
===========================================

.. currentmodule:: biotite

Biological databases are the backbone of computational biology.
The :mod:`biotite.database` subpackage provides interfaces for popular
online databases like the RCSB PDB or the NCBI Entrez database.

Fetching structure files from the RCSB PDB
------------------------------------------

.. currentmodule:: biotite.database.rcsb

Downloading structure files from the *RCSB PDB* is quite easy:
Simply specify the PDB ID, the file format and the target directory
for the :func:`fetch()` function and you are done.
The function even returns the path to the downloaded file, so you
can just load it via the other *Biotite* subpackages
(more on this later).
We will download on a protein structure of the miniprotein *TC5b*
(PDB: 1L2Y) into a temporary directory.
"""

from os.path import relpath
import biotite
import biotite.database.rcsb as rcsb
file_path = rcsb.fetch("1l2y", "pdb", biotite.temp_dir())
print(relpath(file_path))

########################################################################
# In case you want to download multiple files, you are able to specify a
# list of PDB IDs, which in return gives you a list of file_paths.

# Download files in the more modern mmCIF format
file_paths = rcsb.fetch(["1l2y", "1aki"], "cif", biotite.temp_dir())
print([relpath(file_path) for file_path in file_paths])

########################################################################
# By default :func:`fetch()` checks whether the file to be fetched
# already exists in the directory, and downloads it, if it does not
# exist yet.
# If you want to download files irrespectively, set :obj:`overwrite` to
# true.

# Download file in the fast and small binary MMTF format
file_path = rcsb.fetch("1l2y", "mmtf", biotite.temp_dir(), overwrite=True)

########################################################################
# In many cases you are not interested in a specific structure, but you
# want a set of structures that fits your desired criteria.
# For this purpose the *RCSB* SEARCH service can be interfaced.
# 
# At first you have to create :class:`Query` object for the property you
# want to filter.
# The :func:`search()` method takes the :class:`Query` and returns a
# list of PDB IDs, which itself can be used as inpt for :func:`fetch()`.

query = rcsb.ResolutionQuery(0.0, 0.6)
pdb_ids = rcsb.search(query)
print(pdb_ids)
files = rcsb.fetch(pdb_ids, "mmtf", biotite.temp_dir())

########################################################################
# Not all query types of the SEARCH service are supported yet. But it is
# quite easy to implement your needed query type by inheriting
# :class:`SimpleQuery`.
# 
# Multiple :class:`SimpleQuery` objects can be 'and'/'or' combined using
# a :class:`CompositeQuery`.

query1 = rcsb.ResolutionQuery(0.0, 1.0)
query2 = rcsb.MolecularWeightQuery(10000, 100000)
composite = rcsb.CompositeQuery("and", [query1, query2])

########################################################################
# Fetching files from the NCBI Entrez database
# --------------------------------------------
# 
# .. currentmodule:: biotite.database.entrez
# 
# Another important source of biological information is the
# *NCBI Entrez* database, which is commonly known as *the NCBI*.
# It provides a myriad of information, ranging from sequences and
# sequence features to scientific papers. Fetching files from
# NCBI Entrez works analogous to the RCSB interface. This time
# we have to provide the UIDs (Accession or GI) instead of PDB IDs
# to the :func:`fetch()` function.
# Furthermore, we need to specifiy the database to retrieve the data
# from and the retrieval type.

from os.path import relpath
import biotite
import biotite.database.entrez as entrez
# Fetch a single UID ...
file_path = entrez.fetch(
    "NC_001416", biotite.temp_dir(), suffix="fa",
    db_name="nuccore", ret_type="fasta"
)
print(relpath(file_path))
# ... or multiple UIDs 
file_paths = entrez.fetch(
    ["1L2Y_A","1AKI_A"], biotite.temp_dir(), suffix="fa",
    db_name="protein", ret_type="fasta"
)
print([relpath(file_path) for file_path in file_paths])

########################################################################
# A list of valid database, retrieval type and mode combinations can
# be found
# `here <https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly>`_.
# Furthermore :func:`get_database_name()` can be helpful to get the
# required database name by the more commonly known names.

print(entrez.get_database_name("Nucleotide"))

########################################################################
# The *Entrez* database allows for packing data for multiple UIDs into a
# single file. This is achieved with the :func:`fetch_single_file()`
# function.

file_path = entrez.fetch_single_file(
    ["1L2Y_A","1AKI_A"], biotite.temp_file("fa"),
    db_name="protein", ret_type="fasta"
)
print(relpath(file_path))

########################################################################
# Similar to the *RCSB PDB*, you can also search in the *NCBI Entrez*
# database, but in an even more powerful manner:
# Due to the simple design of the search queries accepted by
# *NCBI Entrez*, you can search in every
# `field <https://www.ncbi.nlm.nih.gov/books/NBK49540/>`_.
# of the database.

# Search in all fields
print(entrez.SimpleQuery("BL21 genome"))
# Search in the 'Organism' field
print(entrez.SimpleQuery("Escherichia coli", field="Organism"))

########################################################################
# You can even combine multiple :class:`Query` objects in any way you
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
# the GIs, that can be used as input to :func:`fetch()`.

# Return a maximum number of 10 entries
gis = entrez.search(composite_query, "protein", number=10)
print(gis)