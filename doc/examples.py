# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"
__all__ = ["create_example_file", "create_example_index"]

from os.path import realpath, dirname, join, isdir, isfile, basename
from os import listdir


_indent = " " * 3

def create_example_file(directory):
    lines = []
    
    with open(join(directory, "title"), "r") as f:
        title = f.read().strip()
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")
    
    if isfile(join(directory, "example.png")):
        lines.append(".. image:: example.png")
        lines.append("")
    if isfile(join(directory, "example.txt")):
        lines.append(".. literalinclude:: example.txt")
        lines.append(_indent + ":language: none")
        lines.append("")
    
    lines.append(".. literalinclude:: script.py")
    lines.append("")
    lines.append("(:download:`Source code <script.py>`)")
    lines.append("")
    
    with open(join(directory, "example.rst"), "w") as f:
        f.writelines([line+"\n" for line in lines])


def create_example_index():
    lines = []
    
    lines.append("Examples")
    lines.append("=" * len("Examples"))
    lines.append("")
    
    lines.append(".. toctree::")
    lines.append("")
    dirs = listdir("examples")
    for d in dirs:
        if isdir(join("examples", d)):
            lines.append(_indent + join(basename(d), "example"))
    with open("examples/index.rst", "w") as f:
        f.writelines([line+"\n" for line in lines])