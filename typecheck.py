# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import glob
import re
import mypy.api as mypy

def module_from_file(file_name):
    return file_name.replace("src/", "") \
                    .replace(".pyi", "") \
                    .replace(".py", "") \
                    .replace("/", ".") \
                    .replace(".__init__", "")


def print_errors(err_dict):
    for module, errors in err_dict.items():
        print(module)
        for err in errors:
            line = err[0]
            msg = err[1]
            print(f"{line}:\t{msg}")
        print()


ignore = [
    "Name '.*' already defined \(by an import\)"
]
ignore_patterns = [re.compile(e) for e in ignore]
py_files = glob.glob("src/biotite/**/*.py", recursive=True)
pyi_files = glob.glob("src/biotite/**/*.pyi", recursive=True)
py_dict = {}
pyi_dict = {}

for file in py_files:
    out, errors, status = mypy.run(["--ignore-missing-imports", file])
    for err in out.split("\n"):
        if err != "":
            err = err.split(":")
            file_name = err[0].strip()
            line = err[1].strip()
            err_type = err[2].strip()
            msg = err[3].strip()
            is_stub = (".pyi" in file_name)
            module = module_from_file(file_name)
            if err_type == "error":
                is_ignored = False
                for pattern in ignore_patterns:
                    if pattern.match(msg) is not None:
                        is_ignored = True
                        break
                if not is_ignored:
                    if is_stub:
                        err_dict = pyi_dict
                    else:
                        err_dict = py_dict
                    if module not in err_dict:
                        err_dict[module] = []
                    if (line, msg) not in err_dict[module]:
                        err_dict[module].append((line, msg))


print("Code:")
print_errors(py_dict)
print()
print()
print("Stubs:")
print_errors(pyi_dict)