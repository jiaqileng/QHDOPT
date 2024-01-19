import numpy as np

from qhdopt import QHD


def get_function_from_qp_file(dimension):
    with open(f"{dimension}d_instance.npy", 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
    return QHD.quad_to_gen(Q, b)


def TTS(t_f, p_s, success_prob=0.99):
    p_s = min(p_s, success_prob)
    return t_f * (np.log(1 - success_prob) / (np.log(1 - p_s)))


def calc_success_prob(best, samples, num_shots, f, tol):
    number_success = 0
    for sample in samples:
        if sample is not None and abs(f(sample).item() - best) < tol:
            number_success += 1
    return number_success / num_shots
