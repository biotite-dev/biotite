# Copyright 2017 Patrick Kunzmann.
# This code is part of the Biopython distribution and governed by its
# license.  Please see the LICENSE file that should have been included
# as part of this package.

from ...features import *

__all__ = ["get_feature", "feature_builder"]


_builders = {}


def get_feature(table_entry):
    key, locs, qual_dict = table_entry
    try:
        func = _builders[key]
    except KeyError:
        raise ValueError("Invalid feature key '{:}'".format(key))
    return func(locs, qual_dict)


def feature_builder(feature_key):
    def decorator(func):
        _builders[feature_key] = func
        return func
    return decorator


@feature_builder("CDS")
def build_cds(locs, qual):
    if "gene" in qual:
        gene = qual["gene"]
    else:
        gene = None
    if "product" in qual:
        product = qual["product"]
    else:
        product = "hypothetical protein"
    return CDSFeature(qual["product"], gene, locs)