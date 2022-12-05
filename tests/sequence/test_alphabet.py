# This source code is part of the Biotite package and is distributed
# under the 3-Clause BSD License. Please see 'LICENSE.rst' for further
# information.

import itertools
import pytest
import numpy as np
import biotite.sequence as seq


test_cases = {
    "A" : [0],
    "D" : [3],
    "ABC" : [0,1,2,],
    "ABAFF" : [0,1,0,5,5]
}


@pytest.fixture
def alphabet_symbols():
    return "ABCDEF"


@pytest.mark.parametrize(
    "symbols, exp_code, use_letter_alphabet",
    zip(
        list(test_cases.keys()  ) * 2,
        list(test_cases.values()) * 2,
        [False] * len(test_cases) + [True] * len(test_cases)
    )
)
def test_encoding(alphabet_symbols, symbols, exp_code, use_letter_alphabet):
    if use_letter_alphabet:
        alph = seq.LetterAlphabet(alphabet_symbols)
    else:
        alph = seq.Alphabet(alphabet_symbols)
    
    if len(symbols) == 1:
        assert alph.encode(symbols[0]) == exp_code[0]
    else:
        assert list(alph.encode_multiple(symbols)) == list(exp_code)


@pytest.mark.parametrize(
    "exp_symbols, code, use_letter_alphabet",
    zip(
        list(test_cases.keys()  ) * 2,
        list(test_cases.values()) * 2,
        [False] * len(test_cases) + [True] * len(test_cases)
    )
)
def test_decoding(alphabet_symbols, exp_symbols, code, use_letter_alphabet):
    if use_letter_alphabet:
        alph = seq.LetterAlphabet(alphabet_symbols)
    else:
        alph = seq.Alphabet(alphabet_symbols)
    
    code = np.array(code, dtype=np.uint8)
    if len(code) == 1:
        assert alph.decode(code[0]) == exp_symbols[0]
    else:
        assert list(alph.decode_multiple(code)) == list(exp_symbols)


@pytest.mark.parametrize(
    "use_letter_alphabet, is_single_val",
    itertools.product(
        [False, True], [False, True]
    )
)
def test_error(alphabet_symbols, use_letter_alphabet, is_single_val):
    if use_letter_alphabet:
        alph = seq.LetterAlphabet(alphabet_symbols)
    else:
        alph = seq.Alphabet(alphabet_symbols)

    if is_single_val:
        with pytest.raises(seq.AlphabetError):
            alph.encode("G")
        with pytest.raises(seq.AlphabetError):
            alph.encode(42)
        with pytest.raises(seq.AlphabetError):
            alph.decode(len(alphabet_symbols))
        with pytest.raises(seq.AlphabetError):
            alph.decode(-1)
    else:
        with pytest.raises(seq.AlphabetError):
            alph.encode_multiple("G")
        with pytest.raises(seq.AlphabetError):
            alph.encode_multiple([42])
        with pytest.raises(seq.AlphabetError):
            alph.decode_multiple(np.array([len(alphabet_symbols)]))
        with pytest.raises(seq.AlphabetError):
            alph.decode_multiple(np.array([-1]))


@pytest.mark.parametrize(
    "symbols",
    ["ABC", b"ABC", ["A","B","C"],
     np.array(["A","B","C"]), np.array([b"A",b"B",b"C"])]
)
def test_input_types(alphabet_symbols, symbols):
    """
    'LetterAlphabet' handles different input iterable types in different
    ways.
    Assert that all ways work.
    """
    alph = seq.LetterAlphabet(alphabet_symbols)
    code = alph.encode_multiple(symbols)
    conv_symbols = alph.decode_multiple(code)
    
    
    if isinstance(symbols, bytes):
        symbols = symbols.decode("ASCII")
    assert list(conv_symbols) == list(
        [symbol.decode("ASCII") if isinstance(symbol, bytes) else symbol
         for symbol in symbols]
    )


@pytest.mark.parametrize("use_letter_alphabet", [False, True])
def test_length(alphabet_symbols, use_letter_alphabet):
    if use_letter_alphabet:
        alph = seq.LetterAlphabet(alphabet_symbols)
    else:
        alph = seq.Alphabet(alphabet_symbols)
    assert len(alph) == len(alphabet_symbols)


@pytest.mark.parametrize("use_letter_alphabet", [False, True])
def test_contains(alphabet_symbols, use_letter_alphabet):
    if use_letter_alphabet:
        alph = seq.LetterAlphabet(alphabet_symbols)
    else:
        alph = seq.Alphabet(alphabet_symbols)
    assert "D" in alph


@pytest.mark.parametrize(
    "source_alph_symbols, target_alph_symbols", 
    [
        ("A", "AB"),
        (["foo", "bar"], ["bar", "foo", 42]),
        ("ACGT", "AGTC"),
        ("ACGT", "ACGNT"),
        (np.arange(0, 1000), np.arange(999, -1, -1)),
    ]
)
def test_alphabet_mapper(source_alph_symbols, target_alph_symbols):
    CODE_LENGTH = 10000
    source_alph = seq.Alphabet(source_alph_symbols)
    target_alph = seq.Alphabet(target_alph_symbols)
    mapper = seq.AlphabetMapper(source_alph, target_alph)
    
    ref_sequence = seq.GeneralSequence(source_alph)
    np.random.seed(0)
    ref_sequence.code = np.random.randint(
        len(source_alph), size=CODE_LENGTH, dtype=int
    )

    test_sequence = seq.GeneralSequence(target_alph)
    test_sequence.code = mapper[ref_sequence.code]

    assert test_sequence.symbols == ref_sequence.symbols


@pytest.mark.parametrize("alphabets, common_alph", [
    (
        [
            seq.NucleotideSequence.alphabet_amb,
            seq.NucleotideSequence.alphabet_unamb,
        ],
        seq.NucleotideSequence.alphabet_amb
    ),
    (
        [
            seq.NucleotideSequence.alphabet_unamb,
            seq.NucleotideSequence.alphabet_amb,
        ],
        seq.NucleotideSequence.alphabet_amb
    ),
])
def test_common_alphabet(alphabets, common_alph):
    """
    Check if :func:`common_alphabet()` correctly identifies the common
    alphabet in a simple known test cases
    """
    seq.common_alphabet(alphabets) == common_alph



def test_common_alphabet_no_common():
    """
    Check if :func:`common_alphabet()` correctly identifies that no
    common alphabet exists in a simple known test case.
    """
    assert seq.common_alphabet([
        seq.NucleotideSequence.alphabet_unamb,
        seq.ProteinSequence.alphabet
    ]) is None