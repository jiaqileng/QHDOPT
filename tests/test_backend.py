import pytest

from qhdopt import QHD
from qhdopt.backend.dwave_backend import DWaveBackend

def test_backend_performance():
    Q = [[-2, 1],
         [1, -1]]
    bt = [3 / 4, -1 / 4]
    minimum = -.75
    model = QHD.QP(Q, bt)
    response = model.classically_optimize(solver="TNC", verbose=1)
    assert abs(response.minimum - minimum) < 1e-3

    model.qutip_setup(resolution=6, time_discretization=40)
    model.optimize(verbose=1)
    assert abs(response.minimum - minimum) < 1e-3
    model.ionq_setup(resolution=6, api_key='ionq_API_key', time_discretization=5,
                     shots=1000, on_simulator=True)
    ionq_backend = model.compile_only()
    assert len(ionq_backend.qs.evos) == 5

    model.dwave_setup(resolution=2,
                      api_key='dwave_api_key')
    dwave_compile = model.compile_only()
    assert dwave_compile.chain_strength == 0.1875
    assert dwave_compile.penalty_coefficient == 0.140625
    dwave_compile.print_compilation_info()



