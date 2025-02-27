# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__name__ = "biotite.database.entrez"
__author__ = "Patrick Kunzmann"
__all__ = ["get_database_name"]


# fmt: off
_db_names = {
    "BioProject"        : "bioproject",
    "BioSample"         : "biosample",
    "Biosystems"        : "biosystems",
    "Books"             : "books",
    "Conserved Domains" : "cdd",
    "dbGaP"             : "gap",
    "dbVar"             : "dbvar",
    "Epigenomics"       : "epigenomics",
    "EST"               : "nucest",
    "Gene"              : "gene",
    "Genome"            : "genome",
    "GEO Datasets"      : "gds",
    "GEO Profiles"      : "geoprofiles",
    "GSS"               : "nucgss",
    "HomoloGene"        : "homologene",
    "MeSH"              : "mesh",
    "NCBI C++ Toolkit"  : "toolkit",
    "NCBI Web Site"     : "ncbisearch",
    "NLM Catalog"       : "nlmcatalog",
    "Nucleotide"        : "nuccore",
    "OMIA"              : "omia",
    "PopSet"            : "popset",
    "Probe"             : "probe",
    "Protein"           : "protein",
    "Protein Clusters"  : "proteinclusters",
    "PubChem BioAssay"  : "pcassay",
    "PubChem Compound"  : "pccompound",
    "PubChem Substance" : "pcsubstance",
    "PubMed"            : "pubmed",
    "PubMed Central"    : "pmc",
    "SNP"               : "snp",
    "SRA"               : "sra",
    "Structure"         : "structure",
    "Taxonomy"          : "taxonomy",
    "UniGene"           : "unigene",
    "UniSTS"            : "unists"
}
# fmt: on


def get_database_name(database):
    """
    Map a common NCBI Entrez database name to an E-utility database
    name.

    Parameters
    ----------
    database : str
        Entrez database name.

    Returns
    -------
    name : str
        E-utility database name.

    Examples
    --------

    >>> print(get_database_name("Nucleotide"))
    nuccore
    """
    return _db_names[database]


def sanitize_database_name(db_name):
    """
    Map a common NCBI Entrez database name to an E-utility database
    name, return E-utility database name, or raise an exception if the
    database name is not existing.

    Only for internal usage in ``download.py`` and ``query.py``.

    Parameters
    ----------
    db_name : str
        Entrez database name.

    Returns
    -------
    name : str
        E-utility database name.
    """
    if db_name in _db_names.keys():
        # Convert into E-utility database name
        return _db_names[db_name]
    elif db_name in _db_names.values():
        # Is already E-utility database name
        return db_name
    else:
        raise ValueError("Database '{db_name}' is not existing")
