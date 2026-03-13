import pytest
import biotite.sequence as seq

SEQ_LENGTH = 10_000


@pytest.fixture(scope="module")
def alphabet():
    return seq.ProteinSequence.alphabet


@pytest.fixture(scope="module")
def symbols():
    return "ACDEFGHIKLMNPQRSTVWY" * (SEQ_LENGTH // 20)


@pytest.fixture(scope="module")
def code(alphabet, symbols):
    return alphabet.encode_multiple(symbols)


@pytest.mark.benchmark
def benchmark_encode(alphabet, symbols):
    """
    Encode symbols into a sequence code.
    """
    alphabet.encode_multiple(symbols)


@pytest.mark.benchmark
def benchmark_decode(alphabet, code):
    """
    Decode a sequence code into symbols.
    """
    alphabet.decode_multiple(code)
