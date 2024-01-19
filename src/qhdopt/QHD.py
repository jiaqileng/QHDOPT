import random
import time

import cyipopt
import jax.numpy as jnp
import numpy as np
from qhdopt.utils.function_preprocessing_utils import decompose_function, gen_affine_transformation, \
    gen_new_func_with_affine_trans, generate_bounds
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
        affine_transformation = gen_affine_transformation(self.scaling_factor, self.lb)
        func, syms = gen_new_func_with_affine_trans(affine_transformation, self.func, self.syms)
        self.univariate_dict, self.bivariate_dict = decompose_function(func, syms)

    @classmethod
    def SymPy(cls, func, syms, bounds=None):
        return cls(func, syms, bounds)

    @classmethod
    def QP(cls, Q, b, bounds=None):
        f, xl = QHD.quad_to_gen(Q, b)
        return cls(f, xl, bounds)

    @staticmethod
    def quad_to_gen(Q, b):
        x = symbols(f"x:{len(Q)}")
        f = 0
        for i in range(len(Q)):
            qii = Q[i][i]
            bi = b[i]
            f += 0.5 * qii * x[i] * x[i] + bi * x[i]
        for i in range(len(Q)):
            for j in range(i + 1, len(Q)):
                if Q[i][j] != Q[j][i]:
                    raise Exception(
                        "Q matrix is not symmetric."
                    )
                f += Q[i][j] * x[i] * x[j]
        return f, list(x)

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
        x = x.astype(jnp.float32)
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


    def post_process(self):
        if self.qhd_samples is None:
            raise Exception("No results on record.")

        num_samples = len(self.qhd_samples)
        post_qhd_samples = []
        minimizer = np.zeros(self.dimension)
        bounds = Bounds(np.zeros(self.dimension), np.ones(self.dimension))
        current_best = float("inf")
        f_eval_jit = jit(self.f_eval)
        f_eval_grad = jit(grad(f_eval_jit))
        obj_hess = jit(jacrev(jacfwd(f_eval_jit)))
        start_time = time.time()
        for k in range(num_samples):
            if self.qhd_samples[k] is None:
                post_qhd_samples.append(None)
                continue
            x0 = jnp.array(self.qhd_samples[k])
            if self.post_processing_method == "TNC":
                result = minimize(
                    f_eval_jit,
                    x0,
                    method="TNC",
                    jac=f_eval_grad,
                    bounds=bounds,
                    options={"gtol": 1e-6, "eps": 1e-9},
                )
            elif self.post_processing_method == "IPOPT":
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
            post_qhd_samples.append(self.affine_transformation(result.x))
            if self.f_eval(result.x) < current_best:
                current_best = self.f_eval(post_qhd_samples[k])
                minimizer = post_qhd_samples[k]
        end_time = time.time()
        self.post_processed_samples = post_qhd_samples
        self.info["post_processing_time"] = end_time - start_time

        return minimizer, current_best, end_time - start_time

    def calc_success_rate(self):
        succ_cnt = 0
        tol_rate = 0.05
        thres = self.info["refined_minimum"] + tol_rate * (
                1e-5 + abs(self.info["refined_minimum"])
        )
        for x in self.post_processed_samples:
            if x is not None:
                if self.f_eval(x) < thres:
                    succ_cnt += 1
        return succ_cnt / len(self.post_processed_samples)

    def optimize(self, fine_tune=True, verbose=0):
        raw_samples = self.backend.exec(verbose=1, info=self.info)

        self.raw_samples = raw_samples

        coarse_minimizer, coarse_minimum, self.qhd_samples = self.backend.decoder(raw_samples, self.f_eval)
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
            print("Success rate:", self.calc_success_rate())
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
