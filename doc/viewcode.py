# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

"""
Links source code in API reference to corresponding source files and
lines at GitHub.
"""

__author__ = "Patrick Kunzmann"
__all__ = ["linkcode_resolve"]

import ast
import inspect
import re
from enum import Enum, auto
from importlib import import_module
from pathlib import Path
import biotite


class Source(Enum):
    """Type of source file for an attribute."""

    PYTHON = auto()
    CYTHON = auto()
    RUST = auto()


def _index_rust_code(code_lines):
    """
    Find the line position of structs and enums in *Rust* files.

    This analyzer looks for `pub struct` and `pub enum` definitions
    decorated with `#[pyclass]`.

    Parameters
    ----------
    code_lines : list of str
        The *Rust* source code split into lines.

    Returns
    -------
    line_index : dict (str -> tuple(int, int))
        Maps an attribute name to its first and last line in a Rust
        module.
    """
    line_index = {}

    # Track pyclass decorator lines
    pyclass_line = None

    for i, line in enumerate(code_lines):
        stripped_line = line.strip()

        # Skip empty and comment lines
        if len(stripped_line) == 0 or stripped_line.startswith("//"):
            continue

        # Check for #[pyclass] decorator
        if stripped_line.startswith("#[pyclass"):
            pyclass_line = i
            continue

        # Check for pub struct or pub enum after pyclass
        if pyclass_line is not None:
            match = re.match(r"pub\s+(struct|enum)\s+(\w+)", stripped_line)
            if match:
                attr_name = match.group(2)
                attr_line_start = pyclass_line

                # Find the end of the struct/enum by matching braces
                brace_count = 0
                started = False
                attr_line_stop = i + 1

                for j in range(i, len(code_lines)):
                    for char in code_lines[j]:
                        if char == "{":
                            brace_count += 1
                            started = True
                        elif char == "}":
                            brace_count -= 1
                    if started and brace_count == 0:
                        attr_line_stop = j + 1
                        break

                line_index[attr_name] = (
                    # 'One' based indexing
                    attr_line_start + 1,
                    # 'One' based indexing and inclusive stop
                    attr_line_stop,
                )
            pyclass_line = None

    return line_index


def _index_rust_files(rust_src_path):
    """
    Index all Rust source files and their pyclass-decorated attributes.

    Parameters
    ----------
    rust_src_path : Path
        Path to the Rust source directory (src/rust).

    Returns
    -------
    rust_attribute_index : dict(str -> str)
        Maps attribute names to their Rust file paths (relative to src/).
    rust_line_index : dict(str -> tuple(int, int))
        Maps attribute names to their first and last line in the Rust file.
    """
    rust_attribute_index = {}
    rust_line_index = {}

    for file_path in rust_src_path.rglob("*.rs"):
        lines = file_path.read_text().splitlines()
        line_positions = _index_rust_code(lines)
        for attr_name, (first, last) in line_positions.items():
            # Path relative to src/ directory
            rel_path = file_path.relative_to(rust_src_path.parent)
            rust_attribute_index[attr_name] = str(rel_path)
            rust_line_index[attr_name] = (first, last)

    return rust_attribute_index, rust_line_index


def _get_rust_imports(module_path):
    """
    Parse a Python module file to find attributes imported from biotite.rust.

    Parameters
    ----------
    module_path : Path
        Path to the Python module file.

    Returns
    -------
    rust_imports : set of str
        Names of attributes imported from biotite.rust.
    """
    rust_imports = set()

    try:
        tree = ast.parse(module_path.read_text())
    except SyntaxError:
        return rust_imports

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("biotite.rust"):
                for alias in node.names:
                    # Use the local name (asname if aliased, otherwise name)
                    local_name = alias.asname if alias.asname else alias.name
                    rust_imports.add(local_name)

    return rust_imports


def _index_attributes(
    package_name,
    src_path,
    rust_attribute_index=None,
    rust_line_index=None,
):
    """
    Assign a Python module to each combination of (sub)package and
    attribute (e.g. function, class, etc.) in a given (sub)package.

    Parameters
    ----------
    package_name : str
        Name of the (sub)package.
    src_path : Path
        File path to `package_name`.
    rust_attribute_index, rust_line_index : dict or None
        Indices for Rust attributes.
        If None (first call), they are computed from the Rust source files.

    Returns
    -------
    attribute_index : dict( tuple(str, str) -> (str, Source))
        Maps the combination of (sub)package name and attribute to
        the name of a Python module and to the source type.
    extension_line_index : dict( tuple(str, str) -> tuple(int, int) ) )
        Maps the combination of (sub)package name and attribute to
        the first and last line in an extension module (Cython or Rust).
        Does not contain entries for attributes that are not part of an
        extension module.
    """
    if rust_attribute_index is None:
        rust_attribute_index, rust_line_index = _index_rust_files(
            src_path.parent / "rust"
        )

    if not _is_package(src_path):
        # Directory is not a Python package/subpackage
        # -> Nothing to do
        return {}, {}

    attribute_index = {}
    extension_line_index = {}

    # Identify all subdirectories and index them recursively
    for subdir in src_path.iterdir():
        if subdir.is_dir():
            sub_attribute_index, sub_extension_line_index = _index_attributes(
                f"{package_name}.{subdir.name}",
                subdir,
                rust_attribute_index,
                rust_line_index,
            )
            attribute_index.update(sub_attribute_index)
            extension_line_index.update(sub_extension_line_index)

    # Import package
    package = import_module(package_name)

    # Import all modules in directory and index attributes
    source_files = [
        f
        for f in src_path.iterdir()
        if f.is_file() and f.name != "__init__.py" and f.suffix in (".py", ".pyx")
    ]

    for source_file in source_files:
        module_name = f"{package_name}.{source_file.stem}"
        if module_name == "biotite.version":
            # Autogenerated module from hatch-vcs
            # It contains no '__all__' attribute on purpose
            continue
        module = import_module(module_name)

        if not hasattr(module, "__all__"):
            raise AttributeError(f"Module {module_name} has not attribute '__all__'")
        # Only index attributes from modules that are available
        # via respective Biotite (sub-)package
        # If a the attribute is available, the module was imported in
        # the '__init__.py' -> Expect that all attributes from module
        # are available in package
        # For example 'biotite.structure.util' is only used for internal
        # purposes and is not imported in the '__init__.py'
        if not all([hasattr(package, attr) for attr in module.__all__]):
            continue

        # Determine source type
        is_cython = source_file.suffix == ".pyx"
        rust_imports = set() if is_cython else _get_rust_imports(source_file)

        for attribute in module.__all__:
            if attribute in rust_imports and attribute in rust_attribute_index:
                # Attribute is imported from Rust
                source_type = Source.RUST
                rust_file = rust_attribute_index[attribute]
                attribute_index[(package_name, attribute)] = (rust_file, source_type)
                if attribute in rust_line_index:
                    extension_line_index[(package_name, attribute)] = rust_line_index[
                        attribute
                    ]
            elif is_cython:
                source_type = Source.CYTHON
                attribute_index[(package_name, attribute)] = (module_name, source_type)
            else:
                source_type = Source.PYTHON
                attribute_index[(package_name, attribute)] = (module_name, source_type)

        if is_cython:
            lines = source_file.read_text().splitlines()
            for attribute, (first, last) in _index_cython_code(lines).items():
                extension_line_index[(package_name, attribute)] = (first, last)

    return attribute_index, extension_line_index


def _index_cython_code(code_lines):
    """
    Find the line position of classes and functions in *Cython* files.

    This analyzer works in a very simple way:
    It looks for the `def` and `class` keywords at zero-indentation
    level and determines the end of a class/function by the start of the
    next zero-indentation level attribute or the end of the file,
    respectively.

    By the nature of this approach, methods or inner classes are not
    identified.

    Parameters
    ----------
    code_lines : list of str
        The *Cython* source code splitted into lines.

    Returns
    -------
    line_index : dict (str -> tuple(int, int))
        Maps an attribute name to its first and last line in a Cython
        module.
    """
    line_index = {}

    for i in range(len(code_lines)):
        line = code_lines[i]
        stripped_line = line.strip()

        # Skip empty and comment lines
        if len(stripped_line) == 0 or stripped_line[0] == "#":
            continue

        if line.startswith(("def")):
            # Get name of the function:
            # Remove 'def' from line...
            cropped_line = stripped_line[3:].strip()
            # ...and determine the end of the name by finding the
            # subsequent '('
            cropped_line = cropped_line[: cropped_line.index("(")].strip()
            attr_name = cropped_line
        elif line.startswith(("class", "cdef class")):
            cropped_line = stripped_line
            # Get name of the class:
            # Remove potential 'cdef' from line...
            if cropped_line.startswith("cdef"):
                cropped_line = cropped_line[4:].strip()
            # ...and remove 'class' from line...
            cropped_line = cropped_line[5:].strip()
            # ...and determine the end of the name by finding the
            # subsequent '(' or ':'
            index = (
                cropped_line.index("(")
                if "(" in cropped_line
                else cropped_line.index(":")
            )
            cropped_line = cropped_line[:index].strip()
            attr_name = cropped_line
        else:
            # No new attribute -> skip line
            continue

        attr_line_start = i
        attr_line_stop = i + 1
        for j in range(i + 1, len(code_lines)):
            attr_line = code_lines[j]
            if len(attr_line.strip()) == 0 or attr_line.strip()[0] == "#":
                continue
            indent = len(attr_line) - len(attr_line.lstrip())
            if indent == 0:
                # No indentation -> end of attribute
                break
            else:
                # Exclusive stop -> +1
                attr_line_stop = j + 1

        line_index[attr_name] = (
            # 'One' based indexing
            attr_line_start + 1,
            # 'One' based indexing and inclusive stop
            attr_line_stop,
        )

    return line_index


def _is_package(path):
    return (path / "__init__.py").exists()


_attribute_index, _extension_line_index = _index_attributes(
    "biotite",
    Path(__file__).parent.parent / "src" / "biotite",
)


def linkcode_resolve(domain, info):
    version = biotite.__version__
    base_url = f"https://github.com/biotite-dev/biotite/blob/v{version}/src/"

    if domain != "py":
        return None

    package_name = info["module"]
    attr_name = info["fullname"]
    try:
        module_or_path, source_type = _attribute_index[(package_name, attr_name)]
    except KeyError:
        # The attribute is not defined within Biotite
        # It may be e.g. an inherited method from an external source
        return None

    match source_type:
        case Source.RUST:
            if (package_name, attr_name) in _extension_line_index:
                first, last = _extension_line_index[(package_name, attr_name)]
                return base_url + f"{module_or_path}#L{first}-L{last}"
            else:
                return base_url + f"{module_or_path}"

        case Source.CYTHON:
            module_name = module_or_path
            if (package_name, attr_name) in _extension_line_index:
                first, last = _extension_line_index[(package_name, attr_name)]
                return (
                    base_url + f"{module_name.replace('.', '/')}.pyx#L{first}-L{last}"
                )
            else:
                # In case the attribute is not found
                # by the Cython code analyzer
                return base_url + f"{module_name.replace('.', '/')}.pyx"

        case Source.PYTHON:
            module_name = module_or_path
            module = import_module(module_name)

            # Get the object defined by the attribute name,
            # by traversing the 'attribute tree' to the leaf
            obj = module
            for attr_name_part in attr_name.split("."):
                obj = getattr(obj, attr_name_part)

            # Temporarily change the '__module__' attribute, which is set
            # to the subpackage in Biotite, back to the actual module in
            # order to fool Python's inspect module
            obj.__module__ = module_name

            source_lines, first = inspect.getsourcelines(obj)
            last = first + len(source_lines) - 1

            return base_url + f"{module_name.replace('.', '/')}.py#L{first}-L{last}"
