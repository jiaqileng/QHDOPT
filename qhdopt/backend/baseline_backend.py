import numpy as np
import time
from qhdopt.backend.backend import Backend


class BaselineBackend(Backend):
    """
    Backend implementation for doing nothing "quantum" and instead
    just returning uniformly random samples on the lattice points
    """

    def __init__(self,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots, ):
        super().__init__(10, dimension, shots, "onehot", univariate_dict,
                         bivariate_dict)

    def compile(self, info):
        raise NotImplementedError("no compilation for baseline_backend.")

    def exec(self, verbose: int, info: dict, compile_only: bool = False):
        """
        Executing the BaselineBackend which is essentially just returning
        random samples on the lattice points.
        """
        info["compile_time"] = 0

        if verbose > 1:
            self.print_compilation_info()
        if compile_only:
            return

        raw_samples = []
        for _ in range(self.shots):
            sample = [0] * (self.resolution * self.dimension)
            for j in range(self.dimension):
                sample[np.random.randint(self.resolution) + j * self.resolution] = 1
            raw_samples.append(sample)

        info["backend_time"] = 0

        return raw_samples

    def print_compilation_info(self):
        print(f"Number of shots: {self.shots}")
