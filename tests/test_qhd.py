from qhdopt import QHD
import numpy as np
from simuq.dwave import DWaveProvider


def convert_key(tup):
    a, b = tup
    return f"({a}, {b})"



def get_benchmark_h_J(dim):
    with open(f"resources/{dim}d_unary.npy", 'rb') as f:
        h = list(np.load(f, allow_pickle=True))
        J = np.load(f, allow_pickle=True).item()
    return h, J

def get_Q_b(dim):
    with open(f"../examples/resources/{dim}d_instance.npy", 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
    return Q, b

def compare_h_j(h_j_1, h_j_2):
    tol = 1e-3
    h1, j1 = h_j_1
    h2, j2 = h_j_2
    for i in range(len(h1)):
        assert abs(h1[i] - h2[i]) < tol
    assert DWaveProvider.compare_qubo(j1, j2) == 0

def calc_h_J(dim, embedding_scheme="hamming"):
    Q, b = get_Q_b(dim)
    model = QHD.QP(Q, b)
    model.dwave_setup(10, api_key="", embedding_scheme=embedding_scheme, penalty_coefficient=3e-2, chain_strength=0)
    return model.calc_h_and_J()

def qhd_qp_for_dimension(dim):
    with open(f"resources/{dim}d_qubo.npy", 'rb') as f:
        qubo = np.load(f, allow_pickle=True).item()
    h, J = calc_h_J(dim)
    qubo_from_h_j = DWaveProvider.isingToqubo(h, J)
    qubo_from_h_j = {convert_key(key): val for key, val in qubo_from_h_j.items()}
    assert DWaveProvider.compare_qubo(qubo, qubo_from_h_j) == 0

def test_qhd_qp_hamming():
    for dim in [5, 50, 60, 75]:
        qhd_qp_for_dimension(dim)

def test_qhd_qp_unary():
    for dim in [5, 50, 60, 75]:
        compare_h_j(calc_h_J(dim, embedding_scheme="unary"), get_benchmark_h_J(dim))
