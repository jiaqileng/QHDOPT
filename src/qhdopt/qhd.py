import random
import time
import warnings

import cyipopt
import jax.numpy as jnp
import numpy as np
from qhdopt.utils.function_preprocessing_utils import decompose_function, \
    gen_new_func_with_affine_trans, generate_bounds, quad_to_gen
from qhdopt.utils.benchmark_utils import calc_success_prob
from qhdopt.backend import qutip_backend, ionq_backend, dwave_backend, baseline_backend
from jax import grad, jacfwd, jacrev, jit
from scipy.optimize import Bounds, minimize
from sympy import lambdify
import sympy


class QHD:
    def __init__(
            self,
            func,
            syms,
            bounds=None,
    ):
        self.qubits = None
        self.qs = None
        self.univariate_dict = None
        self.bivariate_dict = None
        self.raw_result = None
        self.qhd_samples = None
        self.post_processed_samples = None
        self.info = dict()
        self.syms = syms
        self.syms_index = {syms[i]:i for i in range(len(syms))}
        self.func = func
        self.bounds = bounds
        self.lambda_numpy = lambdify(syms, func, jnp)
        self.dimension = len(syms)
        if len(syms) != len(func.free_symbols):
            warnings.warn("The number of function free symbols does not match the number of syms.",
                          RuntimeWarning)

    def generate_univariate_bivariate_repr(self):
        self.lb, self.scaling_factor = generate_bounds(self.bounds, self.dimension)
        func, syms = gen_new_func_with_affine_trans(self.affine_transformation, self.func,
                                                    self.syms)
        self.univariate_dict, self.bivariate_dict = decompose_function(func, syms)

    @classmethod
    def SymPy(cls, func, syms, bounds=None):
        return cls(func, syms, bounds)

    @classmethod
    def QP(cls, Q, b, bounds=None):
        f, xl = quad_to_gen(Q, b)
        return cls(f, xl, bounds)

    def dwave_setup(
            self,
            resolution,
            shots=100,
            api_key=None,
            api_key_from_file=None,
            embedding_scheme="unary",
            anneal_schedule=None,
            penalty_coefficient=0,
            chain_strength=None,
            penalty_ratio=0.75,
            post_processing_method="TNC",
    ):
        self.generate_univariate_bivariate_repr()
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
        self.generate_univariate_bivariate_repr()
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
        self.generate_univariate_bivariate_repr()
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
        self.shots = shots
        self.post_processing_method = post_processing_method

    def baseline_setup(
            self,
            resolution,
            shots=100,
            embedding_scheme="onehot",
            post_processing_method="TNC",
    ):
        self.generate_univariate_bivariate_repr()
        self.backend = baseline_backend.BaselineBackend(
            resolution=resolution,
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
            embedding_scheme=embedding_scheme,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def affine_transformation(self, x):
        return self.scaling_factor * x + self.lb

    def jax_affine_transformation(self, x):
        return jnp.array(self.scaling_factor) * x + jnp.array(self.lb)

    def f_eval(self, x):
        x = self.jax_affine_transformation(x.astype(jnp.float32))
        return self.lambda_numpy(*x)

    def classically_optimize(self, samples=None, solver="TNC"):
        if samples is None:
            samples = np.random.rand(self.shots, self.dimension)
        num_samples = len(samples)
        opt_samples = []
        minimizer = np.zeros(self.dimension)
        bounds = Bounds(np.zeros(self.dimension), np.ones(self.dimension))
        current_best = float("inf")
        f_eval_jit = jit(self.f_eval)
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
            opt_samples.append(self.affine_transformation(result.x))
            val = float(self.f_eval(result.x))
            if val < current_best:
                current_best = val
                minimizer = opt_samples[k]
        end_time = time.time()

        return opt_samples, minimizer, current_best, end_time - start_time

    def post_process(self):
        if self.qhd_samples is None:
            raise Exception("No results on record.")

        opt_samples, minimizer, current_best, post_processing_time = self.classically_optimize(
            self.qhd_samples, self.post_processing_method)
        self.post_processed_samples = opt_samples
        self.info["post_processing_time"] = post_processing_time

        return minimizer, current_best, post_processing_time

    def optimize(self, fine_tune=True, compile_only=False, verbose=0):
        raw_samples = self.backend.exec(verbose=verbose, info=self.info, compile_only=compile_only)

        if compile_only:
            return

        self.raw_samples = raw_samples

        coarse_minimizer, coarse_minimum, self.qhd_samples = self.backend.decoder(raw_samples,
                                                                                  self.f_eval)
        self.info["coarse_minimum"] = coarse_minimum
        self.info["coarse_minimizer"] = self.affine_transformation(coarse_minimizer)
        self.info["time_end_decoding"] = time.time()

        minimum = coarse_minimum

        self.info["fine_tune_status"] = fine_tune
        if fine_tune:
            refined_minimizer, refined_minimum, _ = self.post_process()
            self.info["refined_minimizer"], self.info["refined_minimum"] = (
                refined_minimizer,
                refined_minimum,
            )
            self.info["refined_minimizer"] = self.affine_transformation(refined_minimizer)
            self.info["time_end_finetuning"] = time.time()

            minimum = refined_minimum

        if verbose > 0:
            self.print_sol_info()
            self.print_time_info()

        return minimum

    def calc_h_and_J(self):
        if not isinstance(self.backend, dwave_backend.DWaveBackend):
            raise Exception(
                "This function is only used for Dwave backends."
            )
        return self.backend.calc_h_and_J()

    def print_sol_info(self):
        print("* Coarse solution")
        print("Minimizer:", self.info["coarse_minimizer"])
        print("Affined Minimizer:", self.info["coarse_minimizer_affined"])
        print("Minimum:", self.info["coarse_minimum"])
        print()

        if self.info["fine_tune_status"]:
            print("* Fine-tuned solution")
            print("Minimizer:", self.info["refined_minimizer"])
            print("Minimum:", self.info["refined_minimum"])
            print("Success rate:",
                  calc_success_prob(self.info["refined_minimum"], self.post_processed_samples,
                                    self.shots, self.f_eval))
            print()

    def print_time_info(self):
        compilation_time = (
                self.info["time_end_compile"] - self.info["time_start_compile"]
        )
        backend_time = self.info["time_end_backend"] - self.info["time_end_compile"]
        decoding_time = self.info["time_end_decoding"] - self.info["time_end_backend"]

        total_runtime = compilation_time + backend_time + decoding_time

        print("* Runtime breakdown")
        print(f"SimuQ compilation: {compilation_time:.3f} s")
        print(f"Backend runtime: {backend_time:.3f} s")
        print(f"Decoding time: {decoding_time:.3f} s")
        if self.info["fine_tune_status"]:
            finetuning_time = (
                    self.info["time_end_finetuning"] - self.info["time_end_decoding"]
            )
            print(f"Fine-tuning time: {finetuning_time:.3f} s")
            total_runtime += finetuning_time

        print(f"* Total time: {total_runtime:.3f} s")

    def get_solution(self, var=None):
        """
        var can be
        - None (return all values)
        - a Symbol (return the value of the symbol)
        - a list of Symbols (return a list of the values of the symbols)
        """
        
        if self.info["fine_tune_status"]:
            values = self.info["refined_minimizer"]
        else:
            values = self.info["coarse_minimizer"]

        if var is None:
            return values
        if isinstance(var, sympy.Symbol):
            return values[self.syms_index[var]]
        # Otherwise, v is a list of Symbols.
        return [values[self.syms_index[v]] for v in var]