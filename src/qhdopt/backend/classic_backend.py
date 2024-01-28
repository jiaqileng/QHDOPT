import numpy as np
import time
from qhdopt.backend.backend import Backend

class ClassicBackend(Backend):
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots,
                 embedding_scheme, ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict)

    def exec(self, verbose, info):
        info["time_start_compile"] = time.time()
        info["time_end_compile"] = time.time()
        
        if self.embedding_scheme != "onehot":
            raise Exception("ClassicBackend only supports onehot encoding.")
        
        raw_samples = []
        for _ in range(self.shots):
            sample = [0] * (self.resolution * self.dimension)
            for j in range(self.dimension):
                sample[np.random.randint(self.resolution) + j * self.resolution] = 1
            raw_samples.append(sample)
        
        info["time_end_backend"] = time.time()
        
        return raw_samples
