from typing import Tuple, List

from simuq import QSystem, Qubit
from simuq.dwave import DWaveProvider
from simuq.transformation import ising_3to2_transform
import numpy as np
import time
from qhdopt.utils.decoding_utils import spin_to_bitstring

from qhdopt.backend.backend import Backend


class DWaveBackend(Backend):
    """
    Backend implementation for Dwave. Find more information about Dwave's
    backend here: https://docs.dwavesys.com/docs/latest/c_gs_2.html
    """
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 trivariate_dict,
                 shots=100,
                 api_key=None,
                 api_key_from_file=None,
                 embedding_scheme="unary",
                 anneal_schedule=None,
                 penalty_coefficient=0,
                 penalty_ratio=0.75,
                 chain_strength_ratio=1.05,
                 quad_scheme=None,
                 quad_penalty_ratio=None):
        """
        Args:
            quad_scheme: Method of quadratization; can be "sub", "min_sel", or None.
            quad_penalty_ratio: Ratio used to calculate penalty coefficients for quadratization.
        """
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict, trivariate_dict)

        if anneal_schedule is None:
            anneal_schedule = [[0, 0], [20, 1]]
        self.anneal_schedule = anneal_schedule

        self.api_key = api_key
        if api_key_from_file is not None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()

        self.penalty_ratio = penalty_ratio
        self.chain_strength_ratio = chain_strength_ratio
        if penalty_coefficient != 0:
            self.penalty_coefficient = penalty_coefficient
            self.chain_strength = np.max([5e-2, chain_strength_ratio * penalty_coefficient])
        else:
            self.penalty_coefficient = penalty_ratio * self.H_p_max_strength if embedding_scheme == "unary" else 0
            chain_strength_multiplier = np.max([1, penalty_ratio])
            self.chain_strength = np.max([5e-2, chain_strength_multiplier * self.H_p_max_strength])

        if len(trivariate_dict) > 0 and quad_scheme is None:
            raise ValueError("quad_scheme must be specified to deal with "
                             "trivaraite terms in the objective function.")
        self.quad_scheme = quad_scheme
        if self.quad_scheme == "sub":
            if quad_penalty_ratio is None:
                raise ValueError("quad_penalty_ratio must be specified for 'sub' scheme")
            self.quad_penalty_ratio = quad_penalty_ratio
            self.quad_penalty_coefficient = self.quad_penalty_ratio * self.H_p_max_strength

    def compile(self, info):
        # --- Construct Hamiltonian ---
        H_p = self.H_p(self.qubits, self.univariate_dict,
                        self.bivariate_dict, self.trivariate_dict)
        self.qs.add_evolution(
            H_p + self.penalty_coefficient * self.H_pen(self.qubits), 1)

        # --- Quadratization ---
        if self.quad_scheme is None:
            self.qs_2local, self.qubits_2local = self.qs, self.qubits
        else:
            self.qs_2local, self.qubits_2local = self._quadratization(self.qs)

        self.dwp = DWaveProvider(self.api_key)
        start_compile_time = time.time()
        self.dwp.compile(self.qs_2local, self.anneal_schedule, self.chain_strength)
        end_compile_time = time.time()
        info["compile_time"] = end_compile_time - start_compile_time

    def exec(self, verbose: int, info: dict, compile_only=False) -> List[List[int]]:
        """
        Execute the Dwave quantum backend using the problem description specified in
        self.univariate_dict, self.bivariate_dict and self.trivariate_dict. It uses 
        self.H_p to generate the problem hamiltonian and then uses Simuq's DwaveProvider 
        to run the evolution on Dwave.

        Args:
            verbose: Verbosity level.
            info: Dictionary to store information about the execution.
            compile_only: If True, the function only compiles the problem and does not run it.

        Returns:
            raw_samples: A list of raw samples from the Dwave backend.
        """
        self.compile(info)
        self.print_compilation_info(verbose)

        if verbose > 1:
            print("Submit Task to D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        start_run_time = time.time()
        self.dwave_response = self.dwp.run(shots=self.shots)
        info["backend_time"] = time.time() - start_run_time
        info["average_qpu_time"] = self.dwp.avg_qpu_time
        info["time_on_machine"] = self.dwp.time_on_machine
        info["overhead_time"] = info["backend_time"] - info["time_on_machine"]

        if verbose > 1:
            print("Received Task from D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        if verbose > 0:
            print(f"Backend QPU Time: {info['time_on_machine']}")
            print(f"Overhead Time: {info['overhead_time']}\n")

        self.raw_samples = [spin_to_bitstring(result) for result in self.dwp.results()]

        return self.raw_samples

    def _quadratization(self, qs: QSystem) -> Tuple[QSystem, List[Qubit]]:
        """Quadratize 3-local ZZZ interactions.

        Parameters
        ----------
        qs : QSystem
            quantum system to be transformed

        Returns
        -------
        QSystem, List[Qubit]
            transformed quantum system, along with the list of new qubits
        """
        if self.quad_scheme == "sub":
            self._quad_sub_info = dict()
            qs_2local, qubits_2local = ising_3to2_transform(qs, variant="sub", peek=self._quad_sub_info,
                                                            penalty=self.quad_penalty_coefficient)
        elif self.quad_scheme == "min_sel":
            qs_2local, qubits_2local = ising_3to2_transform(qs, variant="min_sel")
        else:
            raise ValueError("Unknown quad_scheme.")
        return qs_2local, qubits_2local

    def calc_h_and_J(self) -> Tuple[List, dict]:
        """
        Function for debugging to provide h and J which uniquely specify the problem hamiltonian

        Returns:
            h: List of h values
            J: Dictionary of J values
        """
        raise RuntimeError("This method should not be used in the current git branch")

        (
            penalty_coefficient,
            chain_strength,
        ) = self.calc_penalty_coefficient_and_chain_strength()
        self.qs.add_evolution(
            self.S_x(self.qubits) + self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict) + penalty_coefficient * self.H_pen(self.qubits), 1
        )

        dwp = DWaveProvider(self.api_key)
        return dwp.compile(self.qs, self.anneal_schedule, chain_strength)

    def print_compilation_info(self, verbose: int):
        if verbose > 1:
            print("* Compilation information")
            if verbose > 2:
                print("Final Hamiltonian:")
                print("(Feature under development; only the Hamiltonian is meaningful here)")
                print(self.qs)
            print(f"Annealing schedule parameter: {self.anneal_schedule}")
            print(f"Penalty coefficient: {self.penalty_coefficient}")
            print(f"Chain strength: {self.chain_strength}")
            print(f"Number of shots: {self.shots}")

    def print_quadratization_info(self):
        if self.raw_samples is None:
            raise RuntimeError("No samples on record.")
        if self.quad_scheme is None:
            return

        print("Quadratization method:", self.quad_scheme)

        n_comp = len(self.qubits) # num of computational qubits
        n_anc = len(self.qubits_2local) - len(self.qubits) # num of ancillary qubits
        print("Number of computational qubits:", n_comp)
        print("Number of ancillary qubits:", n_anc)
        if n_anc == 0:
            return

        if self.quad_scheme == "sub":
            pair2anc = self._quad_sub_info["pair2anc"]
            i_lst, j_lst, a_lst = [], [], []
            for ij, a in pair2anc.items():
                i, j = ij
                i_lst.append(i)
                j_lst.append(j)
                a_lst.append(a)
            wrong_frac = []
            for bitstring in self.raw_samples:
                bitstring = np.array(bitstring)
                wrong_frac.append(
                    sum(bitstring[i_lst] * bitstring[j_lst] != bitstring[a_lst]) / len(pair2anc))
            wrong_frac = np.array(wrong_frac)
            print("Fraction of wrongly behaved ancillary qubits in each sample: ")
            with np.printoptions(precision=3):
                print(wrong_frac)
        print()