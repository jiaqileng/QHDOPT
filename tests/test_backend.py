import pytest

from qhdopt import QHD
from qhdopt.backend.dwave_backend import DWaveBackend

def test_backend_performance():
    Q = [[-2, 1],
         [1, -1]]
    bt = [3 / 4, -1 / 4]
    sol = -.75
    model = QHD.QP(Q, bt)
    response = model.classically_optimize(solver="TNC", verbose=1)
    assert abs(response.minimum - sol) < 1e-3

    model.qutip_setup(resolution=6, time_discretization=40)
    model.optimize(verbose=1)
    assert abs(response.minimum - sol) < 1e-3
    model.ionq_setup(resolution=6, api_key='ionq_API_key', time_discretization=5,
                     shots=1000, on_simulator=True)
    ionq_backend = model.compile_only()
    assert len(ionq_backend.qs.evos) == 5



