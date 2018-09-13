# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["get_database_name", "fetch", "fetch_single_file"]

import requests
import os.path
import os
import glob


_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

_fetch_url = ("efetch.fcgi?db={:}"
              "&id={:}"
              "&rettype={:}"
              "&retmode={:}"
              "&tool={:}"
              "&mail={:}")


_databases = {"BioProject"        : "bioproject",
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
              "UniSTS"            : "unists"}

def get_database_name(database):
    """
    Map an NCBI Entrez database name to an E-utility database name.
    The returned value can be used for `db_name` parameter in `fetch()`.
    
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
    return _databases[database]


def fetch(uids, target_path, suffix, db_name, ret_type,
          ret_mode="text", overwrite=False, verbose=False, mail=""):
    """
    Download files from the NCBI Entrez database in various formats.
    
    The data for each UID will be fetched into a separate file.
    
    A list of valid database, retrieval type and mode combinations can
    be found under
    `<https://www.ncbi.nlm.nih.gov/books/NBK25499/table/chapter4.T._valid_values_of__retmode_and/?report=objectonly>`_
    
    This function requires an internet connection.
    
    Parameters
    ----------
    uids : str or iterable object of str
        A single *unique identifier* (UID) or a list of UIDs of the
        file(s) to be downloaded .
    target_path : str
        The target directory of the downloaded files.
    suffix : str
        The file suffix of the downloaded files. This value is
        independent of the retrieval type.
    db_name : str:
        E-utility database name.
    ret_type : str
        Retrieval type
    ret_mode : str
        Retrieval mode
    overwrite : bool, optional
        If true, existing files will be overwritten. Otherwise the
        respective file will only be downloaded if the file does not
        exist yet in the specified target directory. (Default: False)
    verbose: bool, optional
        If true, the function will output the download progress.
        (Default: False)
    mail : str, optional
        A mail address that is appended to to HTTP request. This address
        is contacted in case you contact the NCBI server too often.
        This does only work if the mail address is registered.
    
    Returns
    -------
    files : str or list of str
        The file path(s) to the downloaded files.
        If a single string (a single UID) was given in `uids`,
        a single string is returned. If a list (or other iterable
        object) was given, a list of strings is returned.
    
    See also
    --------
    fetch_single_file
    
    Examples
    --------
    
    >>> files = fetch(["1L2Y_A","3O5R_A"], temp_dir(), suffix="fa",
    ...               db_name="protein", ret_type="fasta")
    >>> print(files)
    ['/home/padix/temp/1L2Y_A.fa', '/home/padix/temp/3O5R_A.fa']
    """
    # If only a single UID is present,
    # put it into a single element list
    if isinstance(uids, str):
        uids = [uids]
        single_element = True
    else:
        single_element = False
    # Create the target folder, if not existing
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    file_names = []
    for i, id in enumerate(uids):
        # Verbose output
        if verbose:
            print(f"Fetching file {i+1:d} / {len(uids):d} ({id})...", end="\r")
        # Fetch file from database
        file_name = os.path.join(target_path, id + "." + suffix)
        file_names.append(file_name)
        if not os.path.isfile(file_name) or overwrite == True:
            r = requests.get((_base_url + _fetch_url)
                             .format(db_name, id, ret_type, ret_mode,
                                     "BiotiteClient", mail))
            content = r.text
            if content.startswith(" Error"):
                raise ValueError(content[8:])
            with open(file_name, "w+") as f:
                f.write(content)
    if verbose:
        print("\nDone")
    # If input was a single ID, return only a single path
    if single_element:
        return file_names[0]
    else:
        return file_names


def fetch_single_file(uids, file_name,
                      db_name, ret_type, ret_mode="text", mail=None):
    """
    Almost the same as `fetch()`, but the data for the given UIDs will
    be stored in a single file.
    
    Parameters
    ----------
    uids : iterable object of str
        A list of UIDs of the
        file(s) to be downloaded .
    file_name : str
        The file path, including file name, to the target file.
    db_name : str:
        E-utility database name.
    ret_type : str
        Retrieval type
    ret_mode : str
        Retrieval mode
    mail : str, optional
        A mail address that is appended to to HTML request. This address
        is contacted in case you contact the NCBI server too often.
        This does only work if the mail address is registered.
    
    Returns
    -------
    file : str
        The file name of the downloaded file.
    
    See also
    --------
    fetch
    """
    uid_list_str = ""
    for id in uids:
        uid_list_str += id + ","
    # Remove terminal comma
    uid_list_str = uid_list_str[:-1]
    r = requests.get((_base_url + _fetch_url)
                     .format(db_name, uid_list_str, ret_type, ret_mode,
                             "BiotiteClient", mail))
    content = r.text
    with open(file_name, "w+") as f:
        f.write(content)
    return file_name
