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

tested_urls = {}
def cannot_connect_to(url):
    if url not in tested_urls:
        try:
            urllib.request.urlopen(url)
            tested_urls[url] = False
        except urllib.error.URLError:
            tested_urls[url] = True
    return tested_urls[url]

def cannot_import(module):
    return importlib.util.find_spec(module) is None

def is_not_installed(program):
    return shutil.which(program) is None