import time
from typing import Callable, List, Dict, Union, Optional

from sympy import lambdify, Symbol
import jax.numpy as jnp

from qhdopt.backend import dwave_backend, ionq_backend, qutip_backend, baseline_backend
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import decompose_function


class QHD_Base:
    def __init__(
        self,
        func: Callable,
        syms: List[Symbol],
        info: Dict[str, Union[int, float, str]]
    ):
        """
        Initializes the QHD_Base class.

        Args:
            func: The function to be optimized.
            syms: The list of sympy Symbols representing the variables of the function.
            info: Dictionary to store miscellaneous information about the optimization process.
        """
        self.func = func
        self.syms = syms
        self.dimension = len(syms)
        self.univariate_dict, self.bivariate_dict = decompose_function(self.func, self.syms)
        lambda_numpy = lambdify(syms, func, jnp)
        self.f_eval = lambda x: lambda_numpy(*x)
        self.info = info

    def dwave_setup(
        self,
        resolution: int,
        shots: int = 100,
        api_key: Optional[str] = None,
        api_key_from_file: Optional[str] = None,
        embedding_scheme: str = "unary",
        anneal_schedule: Optional[List[List[int]]] = None,
        penalty_coefficient: float = 0,
        chain_strength: Optional[float] = None,
        penalty_ratio: float = 0.75,
    ) -> None:
        """
        Sets up the D-Wave backend for quantum optimization.

        Args:
            resolution: Resolution for discretizing variable space.
            shots: Number of sampling shots for the D-Wave device.
            api_key: API key for accessing D-Wave services.
            api_key_from_file: Path to a file containing the API key.
            embedding_scheme: Embedding scheme for problem mapping.
            anneal_schedule: Custom annealing schedule.
            penalty_coefficient: Coefficient for penalty terms.
            chain_strength: Strength of chains in embedding.
            penalty_ratio: Ratio of penalty terms in the objective function.
        """
        self.backend = dwave_backend.DWaveBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            anneal_schedule=anneal_schedule,
            penalty_coefficient=penalty_coefficient,
            chain_strength=chain_strength,
            penalty_ratio=penalty_ratio,
        )

    def ionq_setup(
        self,
        resolution: int,
        shots: int = 100,
        api_key: Optional[str] = None,
        api_key_from_file: Optional[str] = None,
        embedding_scheme: str = "onehot",
        penalty_coefficient: float = 0,
        time_discretization: int = 10,
        gamma: float = 5,
        on_simulator: bool = False,
    ) -> None:
        """
        Sets up the IonQ backend for quantum optimization.

        Args:
            resolution: Resolution for discretizing variable space.
            shots: Number of sampling shots for the IonQ device.
            api_key: API key for accessing IonQ services.
            api_key_from_file: Path to a file containing the API key.
            embedding_scheme: Embedding scheme for problem mapping.
            penalty_coefficient: Coefficient for penalty terms.
            time_discretization: Number of time steps for discretization.
            gamma: Coefficient for transverse field in quantum annealing.
            on_simulator: Flag to run on simulator instead of actual device.
        """
        self.backend = ionq_backend.IonQBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            on_simulator=on_simulator,
            gamma=gamma,
        )

    def qutip_setup(
        self,
        resolution: int,
        shots: int = 100,
        embedding_scheme: str = "onehot",
        penalty_coefficient: float = 0,
        time_discretization: int = 10,
        gamma: float = 5
    ) -> None:
        """
        Sets up the QuTiP backend for quantum simulation.

        Args:
            resolution: Resolution for discretizing variable space.
            shots: Number of simulation shots.
            embedding_scheme: Embedding scheme for problem mapping.
            penalty_coefficient: Coefficient for penalty terms.
            time_discretization: Number of time steps for discretization.
            gamma: Coefficient for transverse field in quantum annealing.
        """
        self.backend = qutip_backend.QuTiPBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            gamma=gamma,
        )

    def baseline_setup(
        self,
        shots: int = 100,
    ) -> None:
        """
        Sets up a classical baseline backend.

        Args:
            shots: Number of sampling shots for baseline method.
        """
        self.backend = baseline_backend.BaselineBackend(
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
        )

    def optimize(
            self,
            compile_only: bool = False,
            verbose: int = 0
    ) -> Optional[Response]:
        """
        Executes the optimization process.

        Args:
            compile_only: Flag for compilation without execution.
            verbose: Verbosity level (0, 1, 2 for increasing detail).

        Returns:
            An instance of Response containing optimization results, None if compile_only is True.
        """
        raw_samples = self.backend.exec(verbose=verbose, info=self.info, compile_only=compile_only)

        if compile_only:
            # note that when compile_only is true, this is the result of just compile.
            return raw_samples

        start_time_decoding = time.time()
        coarse_minimizer, coarse_minimum, self.decoded_samples = self.backend.decoder(raw_samples,
                                                                                      self.f_eval)

        end_time_decoding = time.time()
        self.info["decoding_time"] = end_time_decoding - start_time_decoding
        qhd_response = Response(self.info, self.decoded_samples, coarse_minimum, coarse_minimizer)

        return qhd_response
