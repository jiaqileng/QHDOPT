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

    def exec(self, verbose, info):
        (
            penalty_coefficient,
            chain_strength,
        ) = self.calc_penalty_coefficient_and_chain_strength()
        self.qs.add_evolution(
            self.S_x(self.qubits) + self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        dwp = DWaveProvider(self.api_key)
        self.prvd = dwp

        info["time_start_compile"] = time.time()
        dwp.compile(self.qs, self.anneal_schedule, chain_strength, self.shots)
        info["time_end_compile"] = time.time()
        if verbose > 1:
            print("Submit Task to D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        dwp.run(shots=self.shots)
        info["time_end_backend"] = time.time()
        info["average_qpu_time"] = dwp.avg_qpu_time
        info["time_on_machine"] = dwp.time_on_machine
        info["overhead_time"] = info["time_end_backend"] - info["time_end_compile"] - \
                                     info["time_on_machine"]
        if verbose >= 1:
            print("Received Task from D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
            print(f"Backend QPU Time: {info['time_on_machine']}")
            print(f"Overhead Time: {info['overhead_time']}")
            print()

        self.raw_result = dwp.results()
        raw_samples = []
        for i in range(self.shots):
            raw_samples.append(spin_to_bitstring(self.raw_result[i]))

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