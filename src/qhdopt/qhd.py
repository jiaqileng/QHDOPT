import random
import time

import cyipopt
import jax.numpy as jnp
import numpy as np
from qhdopt.utils.function_preprocessing_utils import decompose_function, gen_affine_transformation, \
    gen_new_func_with_affine_trans, generate_bounds, quad_to_gen
from qhdopt.utils.benchmark_utils import calc_success_prob
from qhdopt.backend import qutip_backend, ionq_backend, dwave_backend
from jax import grad, jacfwd, jacrev, jit
from scipy.optimize import Bounds, minimize
from sympy import lambdify, symbols


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
        self.func = func
        self.bounds = bounds
        self.lambda_numpy = lambdify(syms, func, jnp)
        self.dimension = len(func.free_symbols)

    def generate_univariate_bivariate_repr(self):
        self.lb, self.scaling_factor = generate_bounds(self.bounds, self.dimension)
        func, syms = gen_new_func_with_affine_trans(self.affine_transformation, self.func, self.syms)
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
        self.backend = dwave_backend.DwaveBackend(
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
        self.backend = ionq_backend.IonqBackend(
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
        self.backend = qutip_backend.QutipBackend(
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

    def affine_transformation(self, x):
        return self.scaling_factor * x + self.lb

    def f_eval(self, x):
        x = self.affine_transformation(x.astype(jnp.float32))
        return self.lambda_numpy(*x)

    @staticmethod
    def binstr_to_bitstr(s):
        return list(map(int, list(s)))

    @staticmethod
    def spin_to_bitstring(spin_list):
        # spin_list is a dict
        list_len = len(spin_list)
        binary_vec = np.empty((list_len))
        bitstring = []
        for k in np.arange(list_len):
            if spin_list[k] == 1:
                bitstring.append(0)
            else:
                bitstring.append(1)

        return bitstring

    @staticmethod
    def classicly_optimize(f, samples, dimension, solver="TNC", affine_transformation=None):
        if affine_transformation is None:
            affine_transformation = gen_affine_transformation(1, 0)
        num_samples = len(samples)
        opt_samples = []
        minimizer = np.zeros(dimension)
        bounds = Bounds(np.zeros(dimension), np.ones(dimension))
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
            opt_samples.append(affine_transformation(result.x))
            if f(result.x) < current_best:
                current_best = f(opt_samples[k])
                minimizer = opt_samples[k]
        end_time = time.time()

        return opt_samples, minimizer, current_best, end_time - start_time

    def post_process(self):
        if self.qhd_samples is None:
            raise Exception("No results on record.")

        opt_samples, minimizer, current_best, post_processing_time = QHD.classicly_optimize(
            self.f_eval, self.qhd_samples, self.dimension, self.post_processing_method,
            self.affine_transformation)
        self.post_processed_samples = opt_samples
        self.info["post_processing_time"] = post_processing_time

        return minimizer, current_best, post_processing_time

    def optimize(self, fine_tune=True, verbose=0):
        raw_samples = self.backend.exec(verbose=1, info=self.info)

        self.raw_samples = raw_samples

        coarse_minimizer, coarse_minimum, self.qhd_samples = self.backend.decoder(raw_samples,
                                                                                  self.f_eval)
        self.info["coarse_minimizer"], self.info["coarse_minimum"] = (
            coarse_minimizer,
            coarse_minimum,
        )
        self.info["time_end_decoding"] = time.time()

        minimum = coarse_minimum

        self.info["fine_tune_status"] = fine_tune
        if fine_tune:
            refined_minimizer, refined_minimum, _ = self.post_process()
            self.info["refined_minimizer"], self.info["refined_minimum"] = (
                refined_minimizer,
                refined_minimum,
            )
            self.info["time_end_finetuning"] = time.time()

            minimum = refined_minimum

        if verbose > 0:
            self.print_sol_info()
            self.print_time_info()

        return minimum

    def print_sol_info(self):
        print("* Coarse solution")
        print("Minimizer:", self.info["coarse_minimizer"])
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
