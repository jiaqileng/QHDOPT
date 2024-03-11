import numpy as np
def spin_to_bitstring(spin_list):
    # spin_list is a dict
    list_len = len(spin_list)
    binary_vec = np.empty((list_len))
    bitstring = []
    for k in np.arange(list_len):
        if spin_list[k] == 1:
            bitstring.append(0)
        else:
            bitstring.append(1)

    return bitstring

def hamming_bitstring_to_vec(bitstring, d, r):
    sample = np.zeros(d)
    for i in range(d):
        sample[i] = sum(bitstring[i * r: (i + 1) * r]) / r
    return sample

def binstr_to_bitstr(s):
    return list(map(int, list(s)))

def unary_bitstring_to_vec(bitstring, d, r):
    sample = np.zeros(d)

    for i in range(d):
        x_i = bitstring[i * r: (i + 1) * r]

        in_low_energy_subspace = True
        for j in range(r - 1):
            if x_i[j] > x_i[j + 1]:
                in_low_energy_subspace = False

        if in_low_energy_subspace:
            sample[i] = np.mean(x_i)
        else:
            return None

    return sample


def onehot_bitstring_to_vec(bitstring, d, r):
    sample = np.zeros(d)

    for i in range(d):
        x_i = bitstring[i * r: (i + 1) * r]
        if sum(x_i) != 1:
            return None
        else:
            slot = 0
            while slot < r and x_i[slot] == 0:
                slot += 1
            sample[i] = 1 - slot / r

    return sample

def bitstring_to_vec(embedding_scheme, bitstring, d, r):
    if embedding_scheme == "unary":
        return unary_bitstring_to_vec(bitstring, d, r)
    elif embedding_scheme == "onehot":
        return onehot_bitstring_to_vec(bitstring, d, r)
    elif embedding_scheme == "hamming":
        return hamming_bitstring_to_vec(bitstring, d, r)
    else:
        raise Exception("Illegal embedding scheme.")

