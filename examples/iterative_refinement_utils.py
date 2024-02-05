import sys
sys.path.insert(0, '../')
from qhdopt import QHD
import numpy as np
import mpmath


def to_numpy(v):
    return np.array(v.tolist(), dtype=np.float64)
def run_qhd(matrix, vector):
    vector = to_numpy(-1 * vector).squeeze()
    model = QHD.QP(matrix, vector, bounds=(-1, 1))
    model.dwave_setup(20, api_key_from_file='/Users/samuelkushnir/Documents/dwave_api_key.txt', embedding_scheme="hamming")
    model.optimize(verbose=0, fine_tune=False)
    return model.info["coarse_minimizer"]


def IR_QHD(matrix, vector, IRprecision):
  ### Scaled Iterative Refinement for solving a linear system
  ### matrix: coefficient matrix of the linear system
  ### vector: right-hand side vector of the linear system
    nabla             = 1                             # Scaling factor
    rho               = 4                             # Incremental scaling
    d                 = len(matrix)                   # Dimension
    iteration         = 0
    x                 = mpmath.matrix(np.zeros(d))                 # Solution
    r                 = vector                        # Residual
    # con               = np.linalg.cond(matrix)             # Condition number
    res=[]
    while np.linalg.norm(r)>IRprecision:
        c = run_qhd(to_numpy(matrix), nabla*r)
        c = mpmath.matrix(c)
        x = x + (1/nabla)*c                           # Updating solution
        r = vector - matrix * x               # Calculating resisdual
        if np.linalg.norm(r) != 0:
            nabla = min(rho*nabla,1/(np.linalg.norm(r)))  # Updating scaling factor
        else:
            nabla = rho*nabla
        res.append(np.linalg.norm(r))
        iteration=iteration+1
        print(np.linalg.norm(r))
    return res, x