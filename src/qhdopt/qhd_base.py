import time

from sympy import lambdify
import jax.numpy as jnp

from qhdopt.backend import dwave_backend, ionq_backend, qutip_backend, baseline_backend
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import decompose_function


class QHD_Base:
    def __init__(self, func, syms, info):
        self.func = func
        self.syms = syms
        self.dimension = len(syms)
        self.univariate_dict, self.bivariate_dict = decompose_function(self.func, self.syms)
        lambda_numpy = lambdify(syms, func, jnp)
        self.f_eval = lambda x: lambda_numpy(*x)
        self.info = info

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
    ):
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
            resolution,
            shots=100,
            api_key=None,
            api_key_from_file=None,
            embedding_scheme="onehot",
            penalty_coefficient=0,
            time_discretization=10,
            gamma=5,
            on_simulator=False,
    ):
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
            resolution,
            shots=100,
            embedding_scheme="onehot",
            penalty_coefficient=0,
            time_discretization=10,
            gamma=5):
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
            shots=100,
    ):
        self.backend = baseline_backend.BaselineBackend(
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
        )
    def optimize(self, compile_only=False, verbose=0):
        raw_samples = self.backend.exec(verbose=verbose, info=self.info, compile_only=compile_only)

        if compile_only:
            return

        start_time_decoding = time.time()
        coarse_minimizer, coarse_minimum, self.decoded_samples = self.backend.decoder(raw_samples,
                                                                                      self.f_eval)

        end_time_decoding = time.time()
        self.info["decoding_time"] = end_time_decoding - start_time_decoding
        qhd_response = Response(self.info, self.decoded_samples, coarse_minimum, coarse_minimizer)

        return qhd_response