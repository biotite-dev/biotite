# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["full_name", "link_type"]

from os.path import join, dirname, realpath
import msgpack


_info_dir = dirname(realpath(__file__))
# Data is taken from
# ftp://ftp.wwpdb.org/pub/pdb/data/monomers/components.cif
# (2019/01/27)
with open(join(_info_dir, "residue_names.msgpack"), "rb") as file:
    _res_names = msgpack.load(file, raw=False)
with open(join(_info_dir, "link_types.msgpack"), "rb") as file:
    _link_types = msgpack.load(file, raw=False)


def full_name(res_name):
    return _res_names.get(res_name.upper())


def link_type(res_name):
    return _link_types.get(res_name.upper())