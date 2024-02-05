import numpy as np
import time
from qhdopt.backend.backend import Backend

class BaselineBackend(Backend):
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots,
                 embedding_scheme, ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict)

    def exec(self, verbose, info, compile_only=False):
        info["time_start_compile"] = time.time()
        info["time_end_compile"] = time.time()

        if verbose > 1:
            self.print_compilation_info()
        if compile_only:
            return
        
        if self.embedding_scheme != "onehot":
            raise Exception("BaselineBackend only supports onehot encoding.")
        
        raw_samples = []
        for _ in range(self.shots):
            sample = [0] * (self.resolution * self.dimension)
            for j in range(self.dimension):
                sample[np.random.randint(self.resolution) + j * self.resolution] = 1
            raw_samples.append(sample)
        
        info["time_end_backend"] = time.time()
        
        return raw_samples

    def print_compilation_info(self):
        print(f"Number of shots: {self.shots}")
