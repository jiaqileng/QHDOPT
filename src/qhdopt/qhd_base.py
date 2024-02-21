import time

from sympy import lambdify
import jax.numpy as jnp

from qhdopt.backend import dwave_backend, ionq_backend, qutip_backend, baseline_backend
from qhdopt.response import Response
from qhdopt.utils.function_preprocessing_utils import decompose_function


class QHD_Base:
    def __init__(self, func, syms):
        self.func = func
        self.syms = syms
        self.dimension = len(syms)
        self.univariate_dict, self.bivariate_dict = decompose_function(self.func, self.syms)
        lambda_numpy = lambdify(syms, func, jnp)
        self.f_eval = lambda x: lambda_numpy(*x)
        self.info = dict()

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
            shots=100,
            post_processing_method="TNC",
    ):
        self.backend = baseline_backend.BaselineBackend(
            dimension=self.dimension,
            univariate_dict=self.univariate_dict,
            bivariate_dict=self.bivariate_dict,
            shots=shots,
        )
        self.shots = shots
        self.post_processing_method = post_processing_method

    def optimize(self, fine_tune=True, compile_only=False, verbose=0, samples=None):
        raw_samples = self.backend.exec(verbose=verbose, info=self.info, compile_only=compile_only)

        if compile_only:
            return

        start_time_decoding = time.time()
        coarse_minimizer, coarse_minimum, self.decoded_samples = self.backend.decoder(raw_samples,
                                                                                      self.f_eval)

        end_time_decoding = time.time()
        self.info["decoding_time"] = end_time_decoding - start_time_decoding
        qhd_response = Response(self.decoded_samples, coarse_minimum, coarse_minimizer,
                                    self.jax_affine_transformation, self.info)

        if verbose > 0:
            qhd_response.print_time_info()
            qhd_response.print_solver_info()
        self.response = qhd_response
        return qhd_response