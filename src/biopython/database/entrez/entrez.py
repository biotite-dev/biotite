# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

import requests
import os.path
import os
import glob

__all__ = ["get_database_name", "fetch", "fetch_single_file"]


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
    return _databases(database)


def fetch(uids, target_path, suffix, db_name, ret_type,
          ret_mode="text", overwrite=False, verbose=False, mail=""):
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
            print("Fetching file {:d} / {:d} ({:})..."
                  .format(i+1, len(uids), id), end="\r")
        # Fetch file from database
        file_name = os.path.join(target_path, id + "." + suffix)
        file_names.append(file_name)
        if not os.path.isfile(file_name) or overwrite == True:
            r = requests.get((_base_url + _fetch_url)
                             .format(db_name, id, ret_type, ret_mode,
                                     "BiopythonClient", mail))
            content = r.text
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
    uid_list_str = ""
    for id in uids:
        uid_list_str += id + ","
    # Remove terminal comma
    uid_list_str = uid_list_str[:-1]
    r = requests.get((_base_url + _fetch_url)
                     .format(db_name, uid_list_str, ret_type, ret_mode,
                             "BiopythonClient", mail))
    content = r.text
    with open(file_name, "w+") as f:
        f.write(content)
    return file_name
