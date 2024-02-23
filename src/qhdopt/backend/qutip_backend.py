import time
import random

from simuq.qutip import QuTiPProvider

from qhdopt.backend.backend import Backend
import qutip as qtp
import numpy as np

from qhdopt.utils.decoding_utils import binstr_to_bitstr


class QuTiPBackend(Backend):
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots=100,
                 embedding_scheme="onehot",
                 penalty_coefficient=0,
                 time_discretization=10,
                 nsteps=10000,
                 gamma=5, ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict)
        self.penalty_coefficient = penalty_coefficient
        self.time_discretization = time_discretization
        self.gamma = gamma
        self.nsteps = nsteps

    def exec(self, verbose, info, compile_only=False):
        if self.embedding_scheme != "onehot":
            raise Exception("QuTiP simulation must use one-hot embedding.")

        self.qs.add_evolution(self.H_k(), 10)
        phi1 = lambda t: self.gamma / (1 + self.gamma * (t ** 2))
        phi2 = lambda t: self.gamma * (1 + self.gamma * (t ** 2))
        Ht = lambda t: phi1(t) * self.H_k() + phi2(t) * self.H_p(self.qubits, self.univariate_dict,
                                                                 self.bivariate_dict)
        self.qs.add_td_evolution(Ht, np.linspace(0, 1, self.time_discretization))

        qpp = QuTiPProvider()
        self.prvd = qpp

        g = qtp.Qobj([[1], [0]])
        e = qtp.Qobj([[0], [1]])
        initial_state_per_dim = qtp.tensor([e] + [g] * (self.resolution - 1))
        for j in range(self.resolution - 1):
            initial_state_per_dim += qtp.tensor(
                [g] * (j + 1) + [e] + [g] * (self.resolution - 2 - j)
            )
        initial_state_per_dim = initial_state_per_dim / np.sqrt(self.resolution)

        initial_state = qtp.tensor([initial_state_per_dim] * self.dimension)

        start_compile_time = time.time()

        qpp.compile(self.qs, initial_state=initial_state)
        end_compile_time = time.time()
        info["compile_time"] = end_compile_time - start_compile_time


        if verbose > 1:
            self.print_compilation_info()
        if compile_only:
            return

        start_time_backend = time.time()
        qpp.run(nsteps=self.nsteps)
        self.raw_result = qpp.results()
        raw_samples = random.choices(
            list(self.raw_result.keys()), weights=self.raw_result.values(), k=self.shots
        )
        raw_samples = list(map(binstr_to_bitstr, raw_samples))
        end_time_backend = time.time()
        info["backend_time"] = end_time_backend - start_time_backend

        return raw_samples

    def print_compilation_info(self):
        print("* Compilation information")
        print("Hamiltonian evolution:")
        print(self.qs)
        print(f"Number of shots: {self.shots}")
