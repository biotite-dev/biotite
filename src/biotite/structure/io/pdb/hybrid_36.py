"""
               Prototype/reference implementation for
                      encoding and decoding
                     atom serial numbers and
                     residue sequence numbers
                          in PDB files.
PDB ATOM and HETATM records reserve columns 7-11 for the atom serial
number. This 5-column number is used as a reference in the CONECT
records, which also reserve exactly five columns for each serial
number.
With the decimal counting system only up to 99999 atoms can be stored
and uniquely referenced in a PDB file. A simple extension to enable
processing of more atoms is to adopt a counting system with more than
ten digits. To maximize backward compatibility, the counting system is
only applied for numbers greater than 99999. The "hybrid-36" counting
system implemented in this file is:
  ATOM      1
  ...
  ATOM  99999
  ATOM  A0000
  ATOM  A0001
  ...
  ATOM  A0009
  ATOM  A000A
  ...
  ATOM  A000Z
  ATOM  ZZZZZ
  ATOM  a0000
  ...
  ATOM  zzzzz
I.e. the first 99999 serial numbers are represented as usual. The
following atoms use a base-36 system (10 digits + 26 letters) with
upper-case letters. 43670016 (26*36**4) additional atoms can be
numbered this way. If there are more than 43770015 (99999+43670016)
atoms, a base-36 system with lower-case letters is used, allowing for
43670016 additional atoms. I.e. in total 87440031 (99999+2*43670016)
atoms can be stored and uniquely referenced via CONECT records.
The counting system is designed to avoid lower-case letters until the
range of numbers addressable by upper-case letters is exhausted.
Importantly, with this counting system the distinction between
"traditional" and "extended" PDB files becomes evident only if there
are more than 99999 atoms to be stored. Programs that are
updated to support the hybrid-36 counting system will continue to
interoperate with programs that do not as long as there are less than
100000 atoms.
PDB ATOM and HETATM records also reserve columns 23-26 for the residue
sequence number. This 4-column number is used as a reference in other
record types (SSBOND, LINK, HYDBND, SLTBRG, CISPEP), which also reserve
exactly four columns for each sequence number.
With the decimal counting system only up to 9999 residues per chain can
be stored and uniquely referenced in a PDB file. If the hybrid-36
system is adopted, 1213056 (26*36**3) additional residues can be
numbered using upper-case letters, and the same number again using
lower-case letters. I.e. in total each chain may contain up to 2436111
(9999+2*1213056) residues that can be uniquely referenced from the
other record types given above.
The implementation in this file should run with Python 2.6 or higher.
There are no other requirements. Run this script without arguments to
obtain usage examples.
Note that there are only about 60 lines of "real" code. The rest is
documentation and unit tests.
To update an existing program to support the hybrid-36 counting system,
simply replace the existing read/write source code for integer values
with equivalents of the hy36decode() and hy36encode() functions below.
This file is unrestricted Open Source (cctbx.sf.net).
Please send corrections and enhancements to cctbx@cci.lbl.gov .
See also:
    http://cci.lbl.gov/hybrid_36/
    http://www.pdb.org/ "Dictionary & File Formats"
Ralf W. Grosse-Kunstleve, Feb 2007.
"""
from __future__ import absolute_import, division, print_function
try:
    from six.moves import range
    from six.moves import zip
except ImportError:
    pass

digits_upper = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
digits_lower = digits_upper.lower()
digits_upper_values = dict([pair for pair in zip(digits_upper, range(36))])
digits_lower_values = dict([pair for pair in zip(digits_lower, range(36))])

def encode_pure(digits, value):
    "encodes value using the given digits"
    assert value >= 0
    if (value == 0): return digits[0]
    n = len(digits)
    result = []
    while (value != 0):
        rest = value // n
        result.append(digits[value - rest * n])
        value = rest
    result.reverse()
    return "".join(result)

def decode_pure(digits_values, s):
    "decodes the string s using the digit, value associations for each character"
    result = 0
    n = len(digits_values)
    for c in s:
        result *= n
        result += digits_values[c]
    return result

def hy36encode(width, value):
    "encodes value as base-10/upper-case base-36/lower-case base-36 hybrid"
    i = value
    if (i >= 1-10**(width-1)):
        if (i < 10**width):
            return ("%%%dd" % width) % i
        i -= 10**width
        if (i < 26*36**(width-1)):
            i += 10*36**(width-1)
            return encode_pure(digits_upper, i)
        i -= 26*36**(width-1)
        if (i < 26*36**(width-1)):
            i += 10*36**(width-1)
            return encode_pure(digits_lower, i)
    raise ValueError("value out of range.")

def hy36decode(width, s):
    "decodes base-10/upper-case base-36/lower-case base-36 hybrid"
    if (len(s) == width):
        f = s[0]
        if (f == "-" or f == " " or f.isdigit()):
            try: return int(s)
            except ValueError: pass
            if (s == " "*width): return 0
        elif (f in digits_upper_values):
            try: return decode_pure(
                digits_values=digits_upper_values, s=s) - 10*36**(width-1) + 10**width
            except KeyError: pass
        elif (f in digits_lower_values):
            try: return decode_pure(
                digits_values=digits_lower_values, s=s) + 16*36**(width-1) + 10**width
            except KeyError: pass
    raise ValueError("invalid number literal.")
