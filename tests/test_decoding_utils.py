from qhdopt.utils.decoding_utils import spin_to_bitstring, hamming_bitstring_to_vec, \
    binstr_to_bitstr, unary_bitstring_to_vec
import numpy as np

def test_spin_to_bitstring():
    spin_list = [-1, 1, 1, 1, -1, -1]
    assert spin_to_bitstring(spin_list) == [1, 0, 0, 0, 1, 1]

def test_hamming_bitstring_to_vec():
    r = 4
    bitstring = [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]
    sol = [1/2, 0, 1, 3/4]
    np.testing.assert_array_equal(hamming_bitstring_to_vec(bitstring, 4, r), np.array(sol))

def test_binstr_to_bitstr():
    s = [1.8, 3.4, 5.6]
    assert binstr_to_bitstr(s) == [1, 3, 5]

def test_unary_bitstring_to_vec():
    r = 3
    bitstring = [0, 0, 1, 1, 1, 1, 0, 0, 0]
    sol = [1/3, 1, 0]
    np.testing.assert_array_equal(unary_bitstring_to_vec(bitstring, 3, r), np.array(sol))

    assert unary_bitstring_to_vec([1, 0, 0], 1, r) == None