import random
import time
import warnings
from typing import List, Tuple, Union, Type

import cyipopt
import jax.numpy as jnp
import numpy as np

from qhdopt.qhd_base import QHD_Base
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import decompose_function, \
    gen_new_func_with_affine_trans, generate_bounds, quad_to_gen
from qhdopt.utils.benchmark_utils import calc_success_prob
from qhdopt.backend import qutip_backend, ionq_backend, dwave_backend, baseline_backend
from jax import grad, jacfwd, jacrev, jit
from scipy.optimize import Bounds, minimize
from sympy import lambdify
from sympy.core.function import Function
from sympy.core.symbol import Symbol
import sympy


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
    def SymPy(cls, func: Function, syms: List[Symbol], bounds: Union[Tuple, List, None] = None) -> 'QHD':
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
    def QP(cls, Q: List[List[float]], b: List[float], bounds: Union[Tuple, List, None] = None) -> 'QHD':
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
            shots :int = 100,
            api_key: Union[str, None] =None,
            api_key_from_file: Union[str, None] = None,
            embedding_scheme: str ="unary",
            anneal_schedule: Union[List[List[int]], None] = None,
            penalty_coefficient=0,
            chain_strength=None,
            penalty_ratio=0.75,
            post_processing_method="TNC",
    ):
        """

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
            chain_strength=chain_strength,
            penalty_ratio=penalty_ratio,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def ionq_setup(
            self,
            resolution,
            shots=100,
            api_key=None,
            api_key_from_file=None,
            embedding_scheme="onehot",
            penalty_coefficient=0,
            time_discretization=10,
            gamma=5,
            post_processing_method="TNC",
            on_simulator=False,
    ):
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
            resolution,
            shots=100,
            embedding_scheme="onehot",
            penalty_coefficient=0,
            time_discretization=10,
            gamma=5,
            post_processing_method="TNC",
    ):
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
            shots=100,
            post_processing_method="TNC",
    ):
        func, syms = self.generate_affined_func()
        self.qhd_base = QHD_Base(func, syms, self.info)
        self.qhd_base.baseline_setup(
            shots=shots
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def affine_transformation(self, x):
        return self.scaling_factor * x + self.lb

    def jax_affine_transformation(self, x):
        return jnp.array(self.scaling_factor) * x + jnp.array(self.lb)

    def fun_eval(self, x):
        """
        x is in the original box (non-normalized)
        """
        x = x.astype(jnp.float32)
        return self.lambda_numpy(*x)

    def f_eval(self, x):
        x = self.jax_affine_transformation(x.astype(jnp.float32))
        return self.lambda_numpy(*x)

    def generate_guess_in_box(self, shots=1):
        """
        By default, generate a sample with a single guess (shots = 1)
        """
        initial_guess = []
        for _ in range(shots):
            initial_guess.append(self.lb + self.scaling_factor * np.random.rand(self.dimension))

        return initial_guess

    def validate_guess_in_box(self, guesses):
        for guess in guesses:
            for i in range(len(self.lb)):
                lb = self.lb[i]
                ub = self.lb[i] + self.scaling_factor[i]
                assert ub >= guess[i] >= lb

    def classically_optimize(self, verbose=0, initial_guess=None, solver="IPOPT"):
        self.generate_affined_func()
        if initial_guess is None:
            initial_guess = self.generate_guess_in_box()
        self.validate_guess_in_box(initial_guess)
        ub = [self.lb[i] + self.scaling_factor[i] for i in range(len(self.lb))]
        bounds = Bounds(np.array(self.lb), np.array(ub))
        samples, minimizer, minimum, optimize_time = self.classical_optimizer(initial_guess, bounds,
                                                                              solver, self.fun_eval)

        self.info["refined_minimum"] = minimum
        self.info["fine_tuning_time"] = optimize_time
        self.info["decoding_time"] = 0
        self.info["compile_time"] = 0
        self.info["backend_time"] = 0
        self.info["fine_tune_status"] = True

        classical_response = Response(self.info, refined_samples=samples, refined_minimum=minimum,
                                      refined_minimizer=minimizer)

        if verbose > 0:
            classical_response.print_time_info()
            classical_response.print_solver_info()
        self.response = classical_response

        return classical_response

    def classical_optimizer(self, samples, bounds, solver, f):
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

    def post_process(self):
        if self.decoded_samples is None:
            raise Exception("No results on record.")
        samples = self.decoded_samples
        solver = self.post_processing_method

        ub = [self.lb[i] + self.scaling_factor[i] for i in range(len(self.lb))]
        bounds = Bounds(np.array(self.lb), np.array(ub))
        opt_samples, minimizer, current_best, post_processing_time = self.classical_optimizer(
            samples, bounds, solver, self.fun_eval)
        self.post_processed_samples = opt_samples
        self.info["post_processing_time"] = post_processing_time

        return minimizer, current_best, post_processing_time

    def optimize(self, fine_tune=True, compile_only=False, verbose=0):
        response = self.qhd_base.optimize(compile_only, verbose)
        self.decoded_samples = response.samples
        self.info["fine_tune_status"] = fine_tune
        if fine_tune:
            start_time_finetuning = time.time()
            refined_minimizer, refined_minimum, _ = self.post_process()
            end_time_finetuning = time.time()
            self.info["refined_minimum"] = refined_minimum
            self.info["fine_tuning_time"] = end_time_finetuning - start_time_finetuning
            qhd_response = Response(self.info, self.decoded_samples, response.coarse_minimum,
                                    response.coarse_minimizer,
                                    self.post_processed_samples, refined_minimum, refined_minimizer)
        else:
            qhd_response = Response(self.info, self.decoded_samples, response.coarse_minimum,
                                    response.coarse_minimizer)

        if verbose > 0:
            qhd_response.print_time_info()
            qhd_response.print_solver_info()
        self.response = qhd_response

        return qhd_response

    def affine_mapping(self, minimizer, minimum, samples):
        original_minimizer = self.affine_transformation(minimizer)
        original_minimum = minimum
        original_samples = []

        for k in range(len(samples)):
            if samples[k] is None:
                continue
            original_samples.append(self.affine_transformation(samples[k]))

        return original_minimizer, original_minimum, original_samples

    def calc_h_and_J(self):
        if not isinstance(self.qhd_base.backend, dwave_backend.DWaveBackend):
            raise Exception(
                "This function is only used for Dwave backends."
            )

        return self.qhd_base.backend.calc_h_and_J()

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
