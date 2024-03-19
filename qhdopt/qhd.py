import time
import warnings
from typing import List, Tuple, Union, Optional, Callable

try:
    import cyipopt
except ImportError:
    warnings.warn("cyipopt not found: post-processing using IPOPT will not work.")
    CYIPOPT_IMPORTED = False
else:
    CYIPOPT_IMPORTED = True

import jax.numpy as jnp
import numpy as np
import sympy
from jax import grad, jacfwd, jacrev, jit
from scipy.optimize import Bounds, minimize
from sympy import lambdify
from sympy.core.function import Function
from sympy.core.symbol import Symbol

from qhdopt.backend.backend import Backend
from qhdopt.qhd_base import QHD_Base
from qhdopt.backend import dwave_backend
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import gen_new_func_with_affine_trans, \
    generate_bounds, quad_to_gen


class QHD:
    """
    Provides functionality to run Quantum Hamiltonian Gradient Descent as introduced
    by https://arxiv.org/pdf/2303.01471.pdf

    A user should initialize QHD through the use of the functions: QHD.QP and QHD.Sympy
    """

    def __init__(
            self,
            func: Function,
            syms: List[Symbol],
            bounds: Union[Tuple, List, None] = None,
    ):
        self.qubits = None
        self.qs = None
        self.raw_result = None
        self.decoded_samples = None
        self.post_processed_samples = None
        self.info = dict()
        self.syms = syms
        self.syms_index = {syms[i]: i for i in range(len(syms))}
        self.func = func
        self.bounds = bounds
        self.lambda_numpy = lambdify(syms, func, jnp)
        self.dimension = len(syms)
        if len(syms) != len(func.free_symbols):
            warnings.warn("The number of function free symbols does not match the number of syms.",
                          RuntimeWarning)

    def generate_affined_func(self) -> Tuple[Function, List[Symbol]]:
        """
        Internal method for generating a new Sympy function with an
        affine transformation which calculated from the bounds property
        inputted from the user
        """
        self.lb, self.scaling_factor = generate_bounds(self.bounds, self.dimension)
        func, syms = gen_new_func_with_affine_trans(self.affine_transformation, self.func,
                                                    self.syms)
        return func, syms

    @classmethod
    def SymPy(cls, func: Function, syms: List[Symbol],
              bounds: Union[Tuple, List, None] = None) -> 'QHD':
        """
        Initialize QHD with a sympy function and its symbols

        Args:
            func: Sympy function
            syms: List of symbols
            bounds: Bounds of the function

        Returns:
            QHD: An instance of QHD
        """
        return cls(func, syms, bounds)

    @classmethod
    def QP(cls, Q: List[List[float]], b: List[float],
           bounds: Union[Tuple, List, None] = None) -> 'QHD':
        """
        Initialize QHD with a quadratic programming format

        Args:
            Q: Quadratic matrix
            b: Linear vector
            bounds: Bounds of the function

        Returns:
            QHD: An instance of QHD
        """
        f, xl = quad_to_gen(Q, b)
        return cls(f, xl, bounds)

    def dwave_setup(
            self,
            resolution: int,
            shots: int = 100,
            api_key: Optional[str] = None,
            api_key_from_file: Optional[str] = None,
            embedding_scheme: str = "unary",
            anneal_schedule: Optional[List[List[int]]] = None,
            penalty_coefficient: float = 0,
            penalty_ratio: float = 0.75,
            chain_strength_ratio: float = 1.05,
            post_processing_method: str = "TNC",
            quad_scheme: Optional[str] = None,
            quad_penalty_ratio: Optional[float] = None
    ):
        """
        Configures the settings for quantum optimization using D-Wave systems.

        Args:
            resolution: The number of bits representing each variable.
            shots: The number of times the quantum device runs to find the solution.
            api_key: Direct API key for connecting to D-Wave's cloud service.
            api_key_from_file: Path to a file containing the D-Wave API key.
            embedding_scheme: Method used for mapping logical variables to physical qubits.
            anneal_schedule: Custom annealing schedule for quantum annealing.
            penalty_coefficient: Coefficient used to enforce constraints in the quantum model.
            penalty_ratio: Ratio used to calculate penalty coefficients.
            post_processing_method: Classical optimization method used after quantum sampling.
            chain_strength_ratio: Ratio of strength of chains in embedding.
            quad_scheme: Method of quadratization; can be "sub", "min_sel", or None.
            quad_penalty_ratio: Ratio used to calculate penalty coefficients for quadratization.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.dwave_setup(
            resolution=resolution,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            anneal_schedule=anneal_schedule,
            penalty_coefficient=penalty_coefficient,
            penalty_ratio=penalty_ratio,
            chain_strength_ratio=chain_strength_ratio,
            quad_scheme=quad_scheme,
            quad_penalty_ratio=quad_penalty_ratio
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

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
            post_processing_method: str = "TNC",
            on_simulator: bool = False,
    ):
        """
        Configures the settings for running QHD using IonQ systems.

        Args:
            resolution: The resolution of the quantum representation.
            shots: Number of measurements to perform on the quantum state.
            api_key: API key for accessing IonQ's quantum computing service.
            api_key_from_file: Path to file containing the API key for IonQ.
            embedding_scheme: Strategy for encoding problem variables into quantum states.
            penalty_coefficient: Multiplier for penalty terms in the quantum formulation.
            time_discretization: Number of time steps for simulating quantum evolution.
            gamma: Scaling factor for the quantum evolution's time discretization.
            post_processing_method: Algorithm for refining quantum results classically.
            on_simulator: Flag to indicate if the quantum simulation should run on a simulator.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.ionq_setup(
            resolution=resolution,
            shots=shots,
            api_key=api_key,
            api_key_from_file=api_key_from_file,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            gamma=gamma,
            on_simulator=on_simulator
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def qutip_setup(
            self,
            resolution: int,
            shots: int = 100,
            embedding_scheme: str = "onehot",
            penalty_coefficient: float = 0,
            time_discretization: int = 10,
            gamma: float = 5,
            post_processing_method: str = "TNC",
    ):
        """
        Configures the settings for quantum simulation of QHD using QuTiP.

        Args:
            resolution: The resolution for encoding variables into quantum states.
            shots: Number of repetitions for the quantum state measurement.
            embedding_scheme: Encoding strategy for representing problem variables.
            penalty_coefficient: Coefficient for penalties in the quantum problem formulation.
            time_discretization: Number of intervals for the quantum evolution simulation.
            gamma: Parameter controlling the strength of the quantum system evolution.
            post_processing_method: Classical method used for post-processing quantum results.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.qutip_setup(
            resolution=resolution,
            shots=shots,
            embedding_scheme=embedding_scheme,
            penalty_coefficient=penalty_coefficient,
            time_discretization=time_discretization,
            gamma=gamma,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def baseline_setup(
            self,
            shots: int = 100,
            post_processing_method: str = "TNC",
    ):
        """
        Sets up the baseline configuration for classical optimization comparison.

        Args:
            shots: The number of solution samples to generate.
            post_processing_method: The classical optimization algorithm to use.
        """
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.baseline_setup(
            shots=shots
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def affine_transformation(self, x: np.ndarray) -> np.ndarray:
        """
        Applies an affine transformation to the input array.

        Args:
            x: The input array to be transformed.

        Returns:
            Transformed array according to the defined scaling factor and lower bounds.
        """
        return self.scaling_factor * x + self.lb

    def fun_eval(self, x: np.ndarray):
        """
        function evaluation when x is in the original box (non-normalized)
        """
        x = x.astype(jnp.float32)
        return self.lambda_numpy(*x)

    def generate_guess_in_box(self, shots: int = 1) -> List[np.ndarray]:
        """
        Generates initial guesses within the defined bounds.
        By default, generate a sample with a single guess (shots = 1)

        Args:
            shots: Number of guesses to generate.

        Returns:
            A list containing the generated guesses.
        """
        initial_guess = []
        for _ in range(shots):
            initial_guess.append(self.lb + self.scaling_factor * np.random.rand(self.dimension))

        return initial_guess

    def validate_guess_in_box(self, guesses: List[np.ndarray]) -> None:
        """
        Validates if the provided guesses are within the bounds.

        Args:
            guesses: List of guesses to validate.
        """
        for guess in guesses:
            for i in range(len(self.lb)):
                lb = self.lb[i]
                ub = self.lb[i] + self.scaling_factor[i]
                assert ub >= guess[i] >= lb

    def classically_optimize(self, verbose=0, initial_guess=None, num_shots=100, solver="IPOPT") -> Response:
        """
        Optimizes a given function classically over a set of samples and within specified bounds.

        Args:
            samples: Initial samples for the optimization.
            bounds: Bounds within which the optimization is to be performed.
            solver: The optimization method

        Returns:
            Response object containing samples, minimum, minimizer, and other info
        """
        self.generate_affined_func()
        if initial_guess is None:
            initial_guess = self.generate_guess_in_box(num_shots)

        self.validate_guess_in_box(initial_guess)
        ub = [self.lb[i] + self.scaling_factor[i] for i in range(len(self.lb))]
        bounds = Bounds(np.array(self.lb), np.array(ub))
        start_time = time.time()
        samples, minimizer, minimum, optimize_time = self.classical_optimizer_helper(initial_guess,
                                                                                     bounds,
                                                                                     solver,
                                                                                     self.fun_eval)
        end_time = time.time()

        self.info["refined_minimum"] = minimum
        self.info["refining_time"] = optimize_time
        self.info["decoding_time"] = 0
        self.info["compile_time"] = 0
        self.info["backend_time"] = 0
        self.info["refine_status"] = True
        self.info["refining_time"] = end_time - start_time

        classical_response = Response(self.info, refined_samples=samples, refined_minimum=minimum,
                                      refined_minimizer=minimizer, func=self.fun_eval)

        if verbose > 0:
            classical_response.print_time_info()
            classical_response.print_solver_info()
        self.response = classical_response

        return classical_response

    def classical_optimizer_helper(self, samples: List[np.ndarray], bounds: Bounds, solver: str,
                                   f: Callable) -> Tuple[
        List[np.ndarray], np.ndarray, float, float]:
        """
        Helper function to optimize a given function classically over a set of samples and within specified bounds.

        Args:
            samples: Initial samples for the optimization.
            bounds: Bounds within which the optimization is to be performed.
            solver: The optimization method
            f: The function to be optimized

        Returns:
            Tuple of samples, minimizer, minimum, and post-processing time
        """
        num_samples = len(samples)
        opt_samples = []
        minimizer = np.zeros(self.dimension)
        current_best = float("inf")
        f_eval_jit = jit(f)
        f_eval_grad = jit(grad(f_eval_jit))
        obj_hess = jit(jacrev(jacfwd(f_eval_jit)))
        start_time = time.time()
        for k in range(num_samples):
            if samples[k] is None:
                opt_samples.append(None)
                continue
            x0 = jnp.array(samples[k])
            if solver == "TNC":
                result = minimize(
                    f_eval_jit,
                    x0,
                    method="TNC",
                    jac=f_eval_grad,
                    bounds=bounds,
                    options={"gtol": 1e-6, "eps": 1e-9},
                )
            elif solver == "IPOPT":
                result = cyipopt.minimize_ipopt(
                    f_eval_jit,
                    x0,
                    jac=f_eval_grad,
                    hess=obj_hess,
                    bounds=bounds,
                    options={"tol": 1e-6, "max_iter": 100},
                )
            else:
                raise Exception(
                    "The Specified Post Processing Method is Not Supported."
                )
            opt_samples.append(result.x)
            val = float(f(result.x))
            if val < current_best:
                current_best = val
                minimizer = result.x
        end_time = time.time()
        post_processing_time = end_time - start_time

        return opt_samples, minimizer, current_best, post_processing_time

    def post_process(self) -> Tuple[np.ndarray, float, float]:
        """
        Private function to post-process the QHD samples returned from a quantum backend.

        Returns:
            Tuple of minimizer, minimum, and post-processing time
        """
        if self.decoded_samples is None:
            raise Exception("No results on record.")
        samples = self.decoded_samples
        solver = self.post_processing_method

        ub = [self.lb[i] + self.scaling_factor[i] for i in range(len(self.lb))]
        bounds = Bounds(np.array(self.lb), np.array(ub))
        opt_samples, minimizer, current_best, post_processing_time = self.classical_optimizer_helper(
            samples, bounds, solver, self.fun_eval)
        self.post_processed_samples = opt_samples
        self.info["post_processing_time"] = post_processing_time

        return minimizer, current_best, post_processing_time

    def compile_only(self) -> Backend:
        return self.qhd_base.compile_only()

    def optimize(self, refine: bool = True, verbose: int = 0) -> Response:
        """
        User-facing function to run QHD on the optimization problem

        Args:
            refine: Flag to indicate if fine-tuning should be performed.
            compile_only: Flag to indicate if only the compilation should be performed.
            verbose: Verbosity level (0, 1, 2 for increasing detail).

        Returns:
            Response object containing samples, minimum, minimizer, and other info
        """
        response = self.qhd_base.optimize(verbose)
        self.coarse_minimizer, self.coarse_minimum, self.decoded_samples = self.affine_mapping(
            response.minimizer, response.minimum, response.samples)
        self.info["refine_status"] = refine
        if refine:
            start_time_finetuning = time.time()
            refined_minimizer, refined_minimum, _ = self.post_process()
            end_time_finetuning = time.time()
            self.info["refined_minimum"] = refined_minimum
            self.info["refining_time"] = end_time_finetuning - start_time_finetuning
            qhd_response = Response(self.info, self.decoded_samples, self.coarse_minimum,
                                    self.coarse_minimizer,
                                    self.post_processed_samples, refined_minimum, refined_minimizer,
                                    self.fun_eval)
        else:
            qhd_response = Response(self.info, self.decoded_samples, self.coarse_minimum,
                                    self.coarse_minimizer, self.fun_eval)

        if verbose > 0:
            qhd_response.print_time_info()
            qhd_response.print_solver_info()
            if isinstance(self.qhd_base.backend, dwave_backend.DWaveBackend):
                self.qhd_base.backend.print_quadratization_info()

        self.response = qhd_response

        return qhd_response

    def affine_mapping(self, minimizer: np.ndarray, minimum: float, samples: List[np.ndarray]) -> Tuple[np.ndarray, float, List[np.ndarray]]:
        """
        Maps the minimizer and samples from the normalized space to the original space.

        Args:
            minimizer: The minimizer in the normalized space.
            minimum: The minimum value of the function.
            samples: The samples in the normalized space.

        Returns:
            Tuple of the minimizer, minimum, and samples in the original space.
        """
        original_minimizer = self.affine_transformation(minimizer)
        original_minimum = minimum
        original_samples = []

        for k in range(len(samples)):
            if samples[k] is None:
                original_samples.append(None)
            else:
                original_samples.append(self.affine_transformation(samples[k]))

        return original_minimizer, original_minimum, original_samples

    def get_solution(self, var=None):
        """
        var can be
        - None (return all values)
        - a Symbol (return the value of the symbol)
        - a list of Symbols (return a list of the values of the symbols)
        """

        values = self.response.minimizer

        if var is None:
            return values
        if isinstance(var, sympy.Symbol):
            return values[self.syms_index[var]]
        # Otherwise, v is a list of Symbols.
        return [values[self.syms_index[v]] for v in var]
    
    def solver_param_diagnose(self):
        if not isinstance(self.qhd_base.backend, dwave_backend.DWaveBackend):
            raise Exception(
                "This function is only used for D-Wave backends."
            )
        
        if self.response.samples is None:
            raise Exception(
                "This function must run after executing QHD on D-Wave."
            )

        h, J, _ = self.qhd_base.backend.exec(verbose=0, info=self.qhd_base.info, compile_only=True)
        hmax = np.max(np.abs(list(h)))
        Jmax = np.max(np.abs(list(J.values())))
        chain_break_fraction = self.qhd_base.backend.dwave_response.record['chain_break_fraction']

        shots_in_subspace = len(self.response.coarse_samples)

        print("***Solver Parameter Diagnosis***")
        print("---Solver Parameters---")
        print(f"hmax = {hmax}, Jmax = {Jmax}")
        print(f"penalty ratio = {self.qhd_base.backend.penalty_ratio}, penalty coefficient = {self.qhd_base.backend.penalty_coefficient}")
        print(f"chain strength ratio = {self.qhd_base.backend.chain_strength_ratio}, chain strength = {self.qhd_base.backend.chain_strength}")
        print("---Solution Stats---")
        print(f"total shots = {self.qhd_base.backend.shots}, shots in subspace = {shots_in_subspace}")
        print(f"median chain break fraction = {np.median(chain_break_fraction)}")

