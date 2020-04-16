# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

from os.path import join, dirname, realpath
import urllib
import importlib
import shutil


def data_dir(subdir):
    return join(dirname(realpath(__file__)), subdir, "data")


### Functions for conditional test skips ###

def cannot_connect_to(url):
    try:
        urllib.request.urlopen(url)
        return False
    except urllib.error.URLError:
        return True

def cannot_import(module):
    return importlib.util.find_spec(module) is None

def is_not_installed(program):
    return shutil.which(program) is None