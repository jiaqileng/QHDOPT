from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider
import numpy as np
import time
from qhdopt.utils.decoding_utils import spin_to_bitstring

from qhdopt.backend.backend import Backend


class DWaveBackend(Backend):
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots=100,
                 api_key=None,
                 api_key_from_file=None,
                 embedding_scheme="unary",
                 anneal_schedule=None,
                 penalty_coefficient=0,
                 chain_strength=None,
                 penalty_ratio=0.75, ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict)
        if anneal_schedule is None:
            anneal_schedule = [[0, 0], [20, 1]]
        self.api_key = api_key
        if api_key_from_file is not None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()
        self.anneal_schedule = anneal_schedule
        self.penalty_coefficient = penalty_coefficient
        self.chain_strength = chain_strength
        self.penalty_ratio = penalty_ratio

    def calc_penalty_coefficient_and_chain_strength(self):
        if self.penalty_coefficient != 0 and self.chain_strength is not None:
            return self.penalty_coefficient, self.chain_strength
        qs = QSystem()
        qubits = [Qubit(qs) for _ in range(len(self.qubits))]
        qs.add_evolution(self.S_x(qubits) + self.H_p(qubits, self.univariate_dict, self.bivariate_dict), 1)
        dwp = DWaveProvider(self.api_key)
        h, J = dwp.compile(qs, self.anneal_schedule)
        max_strength = np.max(np.abs(list(h) + list(J.values())))
        penalty_coefficient = (
            self.penalty_ratio * max_strength if self.embedding_scheme == "unary" else 0
        )
        chain_strength = np.max([5e-2, 0.5 * self.penalty_ratio])
        return penalty_coefficient, chain_strength

    def exec(self, verbose, info, compile_only=False):
        penalty_coefficient, chain_strength = self.calc_penalty_coefficient_and_chain_strength()
        self.penalty_coefficient, self.chain_strength = penalty_coefficient, chain_strength
        self.qs.add_evolution(
            self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        dwp = DWaveProvider(self.api_key)
        self.prvd = dwp

        start_compile_time = time.time()
        dwp.compile(self.qs, self.anneal_schedule, chain_strength, self.shots)
        end_compile_time = time.time()
        info["compile_time"] = end_compile_time - start_compile_time

        if verbose > 1:
            self.print_compilation_info()
        if compile_only:
            return

        if verbose > 1:
            print("Submit Task to D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        dwp.run(shots=self.shots)
        info["backend_time"] = time.time() - end_compile_time
        info["average_qpu_time"] = dwp.avg_qpu_time
        info["time_on_machine"] = dwp.time_on_machine
        info["overhead_time"] = info["backend_time"] - info["time_on_machine"]

        if verbose > 1:
            print("Received Task from D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        if verbose > 0:
            print(f"Backend QPU Time: {info['time_on_machine']}")
            print(f"Overhead Time: {info['overhead_time']}\n")

        raw_samples = [spin_to_bitstring(result) for result in dwp.results()]

        return raw_samples

    def calc_h_and_J(self):
        (
            penalty_coefficient,
            chain_strength,
        ) = self.calc_penalty_coefficient_and_chain_strength()
        self.qs.add_evolution(
            self.S_x(self.qubits) + self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        dwp = DWaveProvider(self.api_key)
        return dwp.compile(self.qs, self.anneal_schedule, chain_strength, self.shots)

    def print_compilation_info(self):
        print("* Compilation information")
        print("Final Hamiltonian:")
        print("(Feature under development; only the Hamiltonian is meaningful here)")
        print(self.qs)
        print(f"Annealing schedule parameter: {self.anneal_schedule}")
        print(f"Penalty coefficient: {self.penalty_coefficient}")
        print(f"Chain strength: {self.chain_strength}")
        print(f"Number of shots: {self.shots}")
