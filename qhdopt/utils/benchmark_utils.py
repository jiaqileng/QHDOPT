import numpy as np

from qhdopt.backend import dwave_backend
from qhdopt.backend.dwave_backend import DWaveBackend
from qhdopt.utils.function_preprocessing_utils import quad_to_gen
import statistics


def get_function_from_qp_file(dimension):
    with open(f"resources/{dimension}d_instance.npy", 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
    return quad_to_gen(Q, b)


def TTS(t_f, p_s, success_prob=0.99):
    p_s = min(p_s, success_prob)
    return t_f * (np.log(1 - success_prob) / (np.log(1 - p_s)))


def calc_h_and_J(qhd):
    if not isinstance(qhd.qhd_base.backend, dwave_backend.DWaveBackend):
        raise Exception(
            "This function is only used for Dwave backends."
        )

    return qhd.qhd_base.backend.calc_h_and_J()

def run_test(model, tol=1e-3):
    data_vector = np.zeros(16)
    # # Run QHD with post-processor = Ipopt
    # response = model.optimize(verbose=1)
    # qhd_ipopt_success_prob = response.get_success_probability()
    # data_vector[0] = model.info["average_qpu_time"]
    # data_vector[1] = model.info["post_processing_time"] / len(model.info["sample_times"])
    # data_vector[12] = statistics.stdev(model.info["sample_times"][1:])
    # data_vector[2] = qhd_ipopt_success_prob
    # data_vector[3] = TTS((data_vector[0] + data_vector[1]), data_vector[2])
    # best = response.minimum
    # # Same QHD samples post-processed by TNC
    # qhd_samples = [sample for sample in response.coarse_samples if sample is not None]
    # response = model.classically_optimize(solver="TNC", initial_guess=qhd_samples)
    #
    # qhd_tnc_success_prob = response.get_success_probability(tol=tol, minimum=best)
    #
    #
    # data_vector[4] = model.info["average_qpu_time"]
    # data_vector[5] = model.info["refining_time"] / len(model.info["sample_times"])
    # data_vector[13] = statistics.stdev(model.info["sample_times"][1:])
    # data_vector[6] = qhd_tnc_success_prob
    # data_vector[7] = TTS((data_vector[4] + data_vector[5]), data_vector[6])

    # Run Ipopt with random init
    response = model.classically_optimize(solver="IPOPT", num_shots=model.shots)
    ipopt_success_prob = response.get_success_probability(tol=tol)
    data_vector[8] = response.info["refining_time"] / len(model.info["sample_times"])
    data_vector[14] = statistics.stdev(model.info["sample_times"][1:])
    data_vector[9] = ipopt_success_prob

    # Run TNC with random init
    response = model.classically_optimize(solver="TNC", num_shots=model.shots)
    tnc_success_prob = response.get_success_probability(tol=tol)

    data_vector[10] = response.info["refining_time"] / len(model.info["sample_times"])
    data_vector[15] = statistics.stdev(model.info["sample_times"][1:])
    data_vector[11] = tnc_success_prob

    return data_vector
