from qhdopt import QHD
import numpy as np
from simuq.dwave import DWaveProvider


def convert_key(tup):
    a, b = tup
    return f"({a}, {b})"

def qhd_qp_for_dimension(dim):
    with open(f"../examples/resources/{dim}d_instance.npy", 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
    with open(f"resources/{dim}d_qubo.npy", 'rb') as f:
        qubo = np.load(f, allow_pickle=True).item()
    model = QHD.QP(Q, b)
    model.dwave_setup(10, api_key="", embedding_scheme="hamming")
    h, J = model.calc_h_and_J()
    qubo_from_h_j = DWaveProvider.isingToqubo(h, J)
    qubo_from_h_j = {convert_key(key): val for key, val in qubo_from_h_j.items()}
    assert DWaveProvider.compare_qubo(qubo, qubo_from_h_j) == 0

def test_qhd_qp_hamming():
    for dim in [60]:
        qhd_qp_for_dimension(dim)
