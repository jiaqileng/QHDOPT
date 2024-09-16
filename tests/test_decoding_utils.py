from qhdopt.utils.decoding_utils import spin_to_bitstring


def test_spin_to_bitstring():
    spin_list = [-1, 1, 1, 1, -1, -1]
    assert spin_to_bitstring(spin_list) == [1, 0, 0, 0, 1, 1]

