import time
import random
from typing import Tuple, List

from qhdopt.backend.backend import Backend
from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider
import qutip as qtp
import numpy as np

from qhdopt.utils.decoding_utils import binstr_to_bitstr

from dimod import ising_to_qubo
from phisolve import PhiMIQP, QIHD, MIQP

import jax
from phisolve.utils.jax_utils import jax_device

class PhiSolveBackend(Backend):
    """
    Backend implementation for PhiSolve.
    """
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots=100,
                 n_steps=10000,
                 ballistic=False,
                 device='cpu',
                 seed=None,
                 dt=0.2,
                 a0=1.0,
                 symplectic_integration=False,
                 slow_a=True,
                 embedding_scheme="unary",
                 penalty_coefficient=0,
                 penalty_ratio=0.75,
                 chain_strength_ratio=0):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict, bivariate_dict)
        self.n_steps = n_steps
        self.ballistic = ballistic
        self.device = device 
        self.seed = seed 
        self.dt = dt 
        self.a0 = a0 
        self.symplectic_integration = symplectic_integration 
        self.slow_a = slow_a
        self.penalty_coefficient = penalty_coefficient
        self.penalty_ratio = penalty_ratio
        self.chain_strength_ratio = chain_strength_ratio
        
        jax.config.update("jax_platforms", jax_device(device))


    def calc_penalty_coefficient_and_chain_strength(self) -> Tuple[float, float]:
        """
        Calculates the penalty coefficient and chain strength using self.penalty_ratio.
        """
        if self.penalty_coefficient != 0:
            chain_strength = np.max([5e-2, self.chain_strength_ratio * self.penalty_coefficient])
            return self.penalty_coefficient, chain_strength
          
        qs = QSystem()
        qubits = [Qubit(qs) for _ in range(len(self.qubits))]
        qs.add_evolution(self.S_x(qubits) + self.H_p(qubits, self.univariate_dict, self.bivariate_dict), 1)
        dwp = DWaveProvider(api_key='')
        h, J = dwp.compile(qs, chain_strength=.0)
        max_strength = np.max(np.abs(list(h) + list(J.values())))
        penalty_coefficient = (
            self.penalty_ratio * max_strength if self.embedding_scheme == "unary" else 0
        )
        # chain_strength = np.max([5e-2, 0.5 * self.penalty_ratio])
        # chain_strength_multiplier = np.max([1, self.penalty_ratio])
        # chain_strength = np.max([5e-2, chain_strength_multiplier * max_strength])
        chain_strength = .0
        return penalty_coefficient, chain_strength

    def compile(self, info, override=None):
        penalty_coefficient, chain_strength = self.calc_penalty_coefficient_and_chain_strength()

        if override is not None:
            # penalty_coefficient, chain_strength = 3.5e-2, 4e-2
            penalty_coefficient, chain_strength = override

        self.penalty_coefficient, self.chain_strength = penalty_coefficient, chain_strength
        self.qs.add_evolution(
            self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        self.dwp = DWaveProvider(api_key='')
        start_compile_time = time.time()
        h, J = self.dwp.compile(self.qs, chain_strength=.0)
        n_vars = len(h)
        h = {i: -h[i] for i in range(n_vars)}
        qubo_dict, _ = ising_to_qubo(h, J)
        Q = np.zeros((n_vars, n_vars))
        for (i, j), v in qubo_dict.items():
            if i == j:
                Q[i, j] = 2 * v
                continue
            Q[i, j] = v
            Q[j, i] = v
        # self.qihd_backend = QIHD(Q=Q, n_binary_vars=n_vars)
        qihd_backend = QIHD(n_shots=self.shots,
                            n_steps=self.n_steps,
                            ballistic=self.ballistic,
                            device=self.device,
                            seed=self.seed,
                            dt=self.dt,
                            a0=self.a0,
                            symplectic_integration=self.symplectic_integration,
                            slow_a=self.slow_a
            )
        self.phiqubo_model = PhiMIQP(
            MIQP(Q=Q, w=np.zeros(n_vars), n_binary_vars=n_vars), 
            backend=qihd_backend,
            )
        end_compile_time = time.time()
        info["compile_time"] = end_compile_time - start_compile_time

    def exec(self, verbose: int, info: dict, compile_only=False, override=None) -> List[List[int]]:
        """
        Execute the Dwave quantum backend using the problem description specified in
        self.univariate_dict and self.bivariate_dict. It uses self.H_p to generate
        the problem hamiltonian and then uses Simuq's DwaveProvider to run the evolution
        on Dwave.

        Args:
            verbose: Verbosity level.
            info: Dictionary to store information about the execution.
            compile_only: If True, the function only compiles the problem and does not run it.

        Returns:
            raw_samples: A list of raw samples from the Dwave backend.
        """
        self.compile(info, override)

        if verbose > 1:
            self.print_compilation_info()

        if verbose > 1:
            print("Initiating PhiSolve:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        start_run_time = time.time()
        # self.dwave_response = self.dwp.run(shots=self.shots)
        self.phisolve_response = self.phiqubo_model.solve()
        
        # raw_samples = self.qihd_backend.generate_samples(backend_params)

        info["backend_time"] = time.time() - start_run_time
        # info["average_qpu_time"] = self.dwp.avg_qpu_time
        # info["time_on_machine"] = self.dwp.time_on_machine
        # info["overhead_time"] = info["backend_time"] - info["time_on_machine"]

        if verbose > 1:
            print("Received results from PhiSolve:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        # if verbose > 0:
        #     print(f"Backend QPU Time: {info['time_on_machine']}")
        #     print(f"Overhead Time: {info['overhead_time']}\n")

        # raw_samples = [spin_to_bitstring(result) for result in self.dwp.results()]

        return self.phisolve_response.samples, self.phisolve_response.sample_counts

    def calc_h_and_J(self) -> Tuple[List, dict]:
        """
        Function for debugging to provide h and J which uniquely specify the problem hamiltonian

        Returns:
            h: List of h values
            J: Dictionary of J values
        """
        (
            penalty_coefficient,
            chain_strength,
        ) = self.calc_penalty_coefficient_and_chain_strength()
        self.qs.add_evolution(
            self.S_x(self.qubits) + self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        dwp = DWaveProvider(api_key='')
        return dwp.compile(self.qs, chain_strength=.0)

    def print_compilation_info(self):
        print("* Compilation information")
        print("Final Hamiltonian:")
        print("(Feature under development; only the Hamiltonian is meaningful here)")
        print(self.qs)
        # print(f"Annealing schedule parameter: {self.anneal_schedule}")
        print(f"Penalty coefficient: {self.penalty_coefficient}")
        # print(f"Chain strength: {self.chain_strength}")
        print(f"Number of shots: {self.shots}")
