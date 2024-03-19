from abc import ABC, abstractmethod
from typing import List, Callable, Tuple

from numpy import ndarray
from simuq import QSystem, hlist_sum, Qubit, TIHamiltonian
import numpy as np

from qhdopt.utils.decoding_utils import bitstring_to_vec


class Backend(ABC):
    """
    Abstract backend class which defines common functions for all backends and
    an abstract function: exec which each backend needs to implement.
    """
    def __init__(self, 
                 resolution, 
                 dimension, 
                 shots, 
                 embedding_scheme, 
                 univariate_dict, 
                 bivariate_dict,
                 trivariate_dict):
        self.resolution = resolution
        self.dimension = dimension
        self.qs = QSystem()
        self.qubits = [Qubit(self.qs, name=f'Q{i}') for i in range(self.dimension * self.resolution)]
        if shots == None :
            shots = 100
        self.shots = shots
        self.embedding_scheme = embedding_scheme
        self.univariate_dict = univariate_dict
        self.bivariate_dict = bivariate_dict
        self.trivariate_dict = trivariate_dict

    def S_x(self, qubits: List[Qubit]) -> TIHamiltonian:
        """
        Generates the hamiltonian S_x hamiltonian as defined
        in https://arxiv.org/pdf/2303.01471.pdf equation (F.38)

        Args:
            qubits: List of qubits

        Returns:
            TIHamiltonian: S_x Hamiltonian
        """
        return hlist_sum([qubit.X for qubit in qubits])

    def unary_penalty(self, k: int, qubits: List[Qubit]):
        """
        Generates the unary penalty hamiltonian on the kth sub-system
        as defined in https://arxiv.org/pdf/2401.08550.pdf
        """
        unary_penalty_sum = lambda k: sum(
            [
                qubits[j].Z * qubits[j + 1].Z
                for j in range(k * self.resolution, (k + 1) * self.resolution - 1)
            ]
        )
        return (
                (-1) * qubits[k * self.resolution].Z
                + qubits[(k + 1) * self.resolution - 1].Z
                - unary_penalty_sum(k)
        )

    def H_pen(self, qubits: List[Qubit]) -> TIHamiltonian:
        """
        Generates the penalty hamiltonian across all dimensions of the problem
        uses unary_penalty as a key subroutine.

        Args:
            qubits: List of qubits

        Returns:
            TIHamiltonian: Penalty Hamiltonian
        """
        if self.embedding_scheme == "hamming":
            return 0
        elif self.embedding_scheme == "unary":
            return hlist_sum(
                [self.unary_penalty(p, qubits) for p in range(self.dimension)]
            )

    def H_p(self, qubits: List[Qubit], univariate_dict: dict, bivariate_dict: dict, trivariate_dict: dict) -> TIHamiltonian:
        """Generate the problem hamiltonian.

        Args:
            qubits: List of qubits
            univariate_dict: Dictionary of univariate terms
            bivariate_dict: Dictionary of bivariate terms
            trivariate_dict: Dictionary of trivariate terms

        Returns:
            TIHamiltonian: Problem Hamiltonian
        """
        return self.H_p_univariate_bivariate(qubits, univariate_dict, bivariate_dict) + \
            self.H_p_trivariate(qubits, trivariate_dict)

    def H_p_univariate_bivariate(self, qubits: List[Qubit], univariate_dict: dict, bivariate_dict: dict) -> TIHamiltonian:
        """Part of the problem Hamiltonian arising from the univariate and bivariate terms, as defined 
        in https://arxiv.org/pdf/2303.01471.pdf (F.24) for the hamming embedding, and modified for the 
        unary and one-hot embedding in ways that can be found in https://arxiv.org/pdf/2401.08550.pdf.

        Args:
            qubits: List of qubits
            univariate_dict: Dictionary of univariate terms
            bivariate_dict: Dictionary of bivariate terms

        Returns:
            TIHamiltonian
        """
        # Encoding of the X operator as defined in https://browse.arxiv.org/pdf/2303.01471.pdf (F.16)
        def Enc_X(k):
            S_z = lambda k: sum(
                [qubits[j].Z for j in range(k * self.resolution, (k + 1) * self.resolution)]
            )
            return (1 / 2) + (-1 / (2 * self.resolution)) * S_z(k)

        def get_ham(d, lmda):
            def n_j(d, j):
                return 0.5 * (
                        qubits[(d - 1) * self.resolution + j].I - qubits[
                    (d - 1) * self.resolution + j].Z
                )

            if self.embedding_scheme == "unary":

                def eval_lmda_unary():
                    eval_points = [i / self.resolution for i in range(self.resolution + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_unary()
                H = eval_lmda[0] * qubits[(d - 1) * self.resolution].I
                for i in range(len(eval_lmda) - 1):
                    H += (eval_lmda[i + 1] - eval_lmda[i]) * n_j(d, self.resolution - i - 1)

                return H

            elif self.embedding_scheme == "onehot":

                def eval_lmda_onehot():
                    eval_points = [i / self.resolution for i in range(1, self.resolution + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_onehot()
                H = 0
                for i in range(len(eval_lmda)):
                    H += eval_lmda[i] * n_j(d, self.resolution - i - 1)

                return H

        H: TIHamiltonian = 0
        for key, value in univariate_dict.items():
            coefficient, lmda = value
            if self.embedding_scheme == "hamming":
                H += coefficient*lmda(Enc_X(key - 1))
            else:
                ham = get_ham(key, lmda)
                H += coefficient * ham

        for key, value in bivariate_dict.items():
            d1, d2 = key
            for term in value:
                coefficient, lmda1, lmda2 = term
                if self.embedding_scheme == "hamming":
                    H += coefficient * lmda1(Enc_X(d1 - 1)) * lmda2(Enc_X(d2 - 1))
                else:
                    H += coefficient * (get_ham(d1, lmda1) * get_ham(d2, lmda2))

        return H

    def H_p_trivariate(self, qubits: List[Qubit], trivariate_dict: dict) -> TIHamiltonian:
        """Part of the problem Hamiltonian arising from the trivariate terms.
        If `trivariate_dict` is empty, return 0.

        Args:
            qubits: List of qubits
            trivariate_dict: Dictionary of trivariate terms

        Returns:
            TIHamiltonian
        """
        if len(trivariate_dict) == 0:
            return 0

        if self.embedding_scheme != "unary":
            raise Exception("Only unary embedding is supported at this moment to deal with trivariate terms.")

        def get_ham(d, lmda):
            def n_j(d, j):
                return 0.5 * (
                        qubits[(d - 1) * self.resolution + j].I - qubits[
                    (d - 1) * self.resolution + j].Z
                )

            def eval_lmda_unary():
                eval_points = [i / self.resolution for i in range(self.resolution + 1)]
                return [lmda(x) for x in eval_points]

            eval_lmda = eval_lmda_unary()
            H = eval_lmda[0] * qubits[(d - 1) * self.resolution].I
            for i in range(len(eval_lmda) - 1):
                H += (eval_lmda[i + 1] - eval_lmda[i]) * n_j(d, self.resolution - i - 1)

            return H

        H: TIHamiltonian = 0

        for key, value in trivariate_dict.items():
            d1, d2, d3 = key
            for term in value:
                coefficient, lmda1, lmda2, lmda3 = term
                H += coefficient * (get_ham(d1, lmda1) * get_ham(d2, lmda2) * get_ham(d3, lmda3))

        return H

    @property
    def H_p_max_strength(self):
        """The largest coefficient (in absolute value) among all terms of the problem Hamiltonian
        """
        if hasattr(self, "_H_p_max_strength"):
            return self._H_p_max_strength

        qs = QSystem()
        qubits = [Qubit(qs) for _ in range(len(self.qubits))]
        qs.add_evolution(self.H_p(qubits, self.univariate_dict, self.bivariate_dict, self.trivariate_dict), 1)

        all_coeff = []
        for prod, c in qs.evos[0][0].ham:
            if 'Z' in list(prod.values()):
                all_coeff.append(c)
        self._H_p_max_strength = np.max(np.abs(all_coeff))

        return self._H_p_max_strength

    def decoder(self, raw_samples: List[int], f_eval: Callable) -> Tuple[ndarray, int, List[ndarray]]:
        """
        decodes the raw samples returned from the backend into samples
        which are the form (a_1, a_2,...,a_d) where d is the dimension
        of the problem and a_j is a number between 0 and 1.

        Args:
            raw_samples: List of raw samples
            f_eval: Function to evaluate the samples

        Returns:
            Tuple: minimizer, minimum, qhd_samples
        """
        qhd_samples = []
        minimizer = np.zeros(self.dimension)
        minimum = float("inf")

        for i in range(len(raw_samples)):
            bitstring = raw_samples[i]
            qhd_samples.append(bitstring_to_vec(self.embedding_scheme, bitstring, self.dimension, self.resolution))
            if qhd_samples[i] is None:
                continue
            new_f = float(f_eval(qhd_samples[i]))
            if new_f < minimum:
                minimum = new_f
                minimizer = qhd_samples[i]

        return minimizer, minimum, qhd_samples

    def H_k(self, qubits: List[Qubit] = None) -> TIHamiltonian:
        if qubits is None:
            qubits = self.qubits
        if self.embedding_scheme == "onehot":

            def onehot_driving_sum(k):
                return sum(
                    [
                        0.5
                        * (
                                qubits[j].X * qubits[j + 1].X
                                + qubits[j].Y * qubits[j + 1].Y
                        )
                        for j in range(k * self.resolution, (k + 1) * self.resolution - 1)
                    ]
                )

            return (-0.5 * self.resolution ** 2) * hlist_sum(
                [onehot_driving_sum(p) for p in range(self.dimension)]
            )
        else:
            return (-0.5 * self.resolution ** 2) * hlist_sum([qubit.X for qubit in qubits])

    def compile(self, info):
        """
        Compiles the problem description into a format that the backend can run.

        Args:
            info: Dictionary to store information about the compilation
        """
        pass

    @abstractmethod
    def exec(self, verbose: int, info: dict):
        """
        Executes the quantum backend to run QHD

        Args:
            verbose: Verbosity level
            info: Dictionary to store information about the execution
        """
        pass
