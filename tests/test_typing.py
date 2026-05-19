# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

__author__ = "Patrick Kunzmann"

import inspect
import pkgutil
from importlib import import_module
from pathlib import Path
import pytest
from beartype.roar import BeartypeCallHintParamViolation
import biotite
import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
from tests.util import data_dir

# Modules under the Biotite namespace that should not be introspected.
_EXCLUDED_MODULES = {
    # Build/install script, not intended to be imported
    "biotite.setup_ccd",
}


@pytest.fixture(scope="module")
def atoms():
    pdbx_file = pdbx.BinaryCIFFile.read(Path(data_dir("structure")) / "1l2y.bcif")
    return pdbx.get_structure(pdbx_file, model=1)


def _is_testable_name(name):
    """
    Whether *name* refers to a member that should be type-checked.

    Single-underscore names are private and skipped.
    Double-underscore names are part of the public protocol and included.
    """
    if name.startswith("__") and name.endswith("__"):
        return True
    return not name.startswith("_")


def _is_in_biotite(obj):
    module = getattr(obj, "__module__", None) or ""
    return module.startswith("biotite")


def _iter_biotite_modules():
    """
    Yield each importable submodule of the ``biotite`` package.

    Submodules whose import fails (e.g. due to missing optional
    dependencies) are silently skipped.
    """
    yield biotite
    for info in pkgutil.walk_packages(
        biotite.__path__,
        prefix=biotite.__name__ + ".",
        onerror=lambda _name: None,
    ):
        if info.name in _EXCLUDED_MODULES:
            continue
        if any(info.name.startswith(excl + ".") for excl in _EXCLUDED_MODULES):
            continue
        try:
            yield import_module(info.name)
        except ImportError:
            continue


def _walk_class_methods(cls):
    """
    Yield ``(name, member, defining_class)`` for every routine defined
    on *cls* or one of its Biotite-owned base classes.

    Members inherited from non-Biotite bases (e.g. :class:`object`) are
    skipped, but private Biotite bases (like ``_AtomArrayBase``) are
    walked so that methods inherited into public subclasses are still
    covered.
    """
    seen_names = set()
    for base in cls.__mro__:
        if base is object:
            continue
        if not _is_in_biotite(base):
            continue
        for name in list(vars(base)):
            if name in seen_names:
                continue
            try:
                member = inspect.getattr_static(base, name)
            except AttributeError:
                continue
            seen_names.add(name)
            # Unwrap classmethod / staticmethod descriptors so that
            # `inspect.signature` reports the underlying function.
            if isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            if inspect.isroutine(member):
                yield name, member, base


def _iter_public_callables():
    """
    Yield ``(test_id, callable)`` pairs for every public function and
    method that is defined inside the ``biotite`` namespace, including
    members of classes implemented in the Rust extension module.

    Each callable is yielded exactly once, even if it is re-exported
    from multiple modules.
    """
    seen = set()
    for module in _iter_biotite_modules():
        for attr_name in dir(module):
            if not _is_testable_name(attr_name):
                continue
            try:
                attr = getattr(module, attr_name)
            except AttributeError:
                continue

            if inspect.isclass(attr) and _is_in_biotite(attr):
                for name, member, base in _walk_class_methods(attr):
                    if not _is_testable_name(name):
                        continue
                    key = (base.__module__, base.__qualname__, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    test_id = f"{base.__module__}.{base.__qualname__}.{name}"
                    yield test_id, member

            elif inspect.isroutine(attr) and _is_in_biotite(attr):
                key = (
                    getattr(attr, "__module__", ""),
                    getattr(attr, "__qualname__", attr_name),
                )
                if key in seen:
                    continue
                seen.add(key)
                yield f"{key[0]}.{key[1]}", attr


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(func, id=test_id)
        # Sorting by the test ID groups callables by subpackage (the test ID's
        # leading dotted path) and, within each subpackage, orders them
        # alphabetically by class / attribute name.
        for test_id, func in sorted(_iter_public_callables(), key=lambda pair: pair[0])
    ],
)
def test_all_annotated(func):
    """
    Every parameter and the return value of the given public callable
    must be type-annotated.
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        pytest.skip(f"Signature of {func!r} is not introspectable")

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.annotation is inspect.Parameter.empty:
            raise ValueError(f"Parameter '{name}' lacks annotation")
    if sig.return_annotation is inspect.Signature.empty:
        raise ValueError("Return value lacks annotation")


def test_type_failure(atoms):
    """
    Calling a function with the wrong argument type must raise a
    :class:`BeartypeException` during the test session.
    """
    with pytest.raises(BeartypeCallHintParamViolation):
        struc.distance(atoms, "not an atoms object")
