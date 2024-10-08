from typing import Tuple

from qhdopt import QHD
import numpy as np

from examples.benchmark_utils import calc_h_and_J


def convert_key(tup: Tuple) -> str:
    a, b = tup
    return f"({a}, {b})"


def get_benchmark_h_J(dim):
    with open(f"./resources/{dim}d_unary.npy", 'rb') as f:
        h = list(np.load(f, allow_pickle=True))
        J = np.load(f, allow_pickle=True).item()
    return h, J


def get_Q_b(dim):
    with open(f"./resources/{dim}d_instance.npy", 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
    return Q, b


def compare_h_j(h_j_1, h_j_2):
    tol = 1e-3
    h1, j1 = h_j_1
    h2, j2 = h_j_2
    for i in range(len(h1)):
        assert abs(h1[i] - h2[i]) < tol
    assert compare_qubo(j1, j2) == 0

def isingToqubo(h, J):
    n = len(h)
    QUBO = {}

    for i in range(n):
        s = 0
        for ii in range(n):
            if (i,ii) in J.keys():
                s += J[(i,ii)]
            if (ii,i) in J.keys():
                s += J[(ii,i)]

        QUBO[(i,i)] = -2 * (h[i] + s)

        for j in range(i+1, n):
            if (i,j) in J.keys() and J[i, j] != 0:
                QUBO[(i,j)] = 4 * J[(i,j)]

    return QUBO


def calc_h_J(dim, embedding_scheme="hamming"):
    Q, b = get_Q_b(dim)
    model = QHD.QP(Q, b)
    model.dwave_setup(10, api_key="", embedding_scheme=embedding_scheme, penalty_coefficient=3e-2)
    return calc_h_and_J(model)

def compare_qubo(q1, q2):
    numDifferent = 0
    if len(q1.keys()) != len(q2.keys()): return False
    for key in q1.keys():
        if abs(q1[key] - q2[key]) > 10**-3:
            numDifferent += 1
    return numDifferent

def qhd_qp_for_dimension(dim):
    with open(f"./resources/{dim}d_qubo.npy", 'rb') as f:
        qubo = np.load(f, allow_pickle=True).item()
    h, J = calc_h_J(dim)
    qubo_from_h_j = isingToqubo(h, J)
    qubo_from_h_j = {convert_key(key): val for key, val in qubo_from_h_j.items()}
    assert compare_qubo(qubo, qubo_from_h_j) == 0


def test_qhd_qp_hamming():
    for dim in [5, 50, 60, 75]:
        qhd_qp_for_dimension(dim)


def test_qhd_qp_unary():
    for dim in [5, 50, 60, 75]:
        compare_h_j(calc_h_J(dim, embedding_scheme="unary"), get_benchmark_h_J(dim))
