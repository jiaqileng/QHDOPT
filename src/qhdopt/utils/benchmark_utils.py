import numpy as np
from qhdopt.backend.dwave_backend import DwaveBackend
from qhdopt.utils.function_preprocessing_utils import quad_to_gen


def get_function_from_qp_file(dimension):
    with open(f"resources/{dimension}d_instance.npy", 'rb') as f:
        Q = np.load(f)
        b = np.load(f)
    return quad_to_gen(Q, b)


def TTS(t_f, p_s, success_prob=0.99):
    p_s = min(p_s, success_prob)
    return t_f * (np.log(1 - success_prob) / (np.log(1 - p_s)))


def calc_success_prob(best, samples, num_shots, f, tol=1e-3):
    number_success = 0
    for sample in samples:
        if sample is not None and abs(f(sample).item() - best) < tol:
            number_success += 1
    return number_success / num_shots


def run_test(model, tol=1e-3):
    from qhdopt.qhd import QHD
    data_vector = np.zeros(12)
    # Run QHD with post-processor = Ipopt
    current_best_qhd = model.optimize(verbose=1)
    qhd_ipopt_success_prob = calc_success_prob(current_best_qhd, model.post_processed_samples,
                                               model.shots, model.f_eval, tol)
    if isinstance(model.backend, DwaveBackend):
        data_vector[0] = model.info["average_qpu_time"]
    data_vector[1] = model.info["post_processing_time"]
    data_vector[2] = qhd_ipopt_success_prob
    data_vector[3] = TTS(data_vector[0] + data_vector[1], data_vector[2])

    # Same QHD samples post-processed by TNC
    model.post_processing_method = "TNC"
    model.post_process()
    qhd_tnc_success_prob = calc_success_prob(current_best_qhd, model.post_processed_samples,
                                             model.shots, model.f_eval, tol)
    if isinstance(model.backend, DwaveBackend):
        data_vector[4] = model.info["average_qpu_time"]
    data_vector[5] = model.info["post_processing_time"]
    data_vector[6] = qhd_tnc_success_prob
    data_vector[7] = TTS(data_vector[4] + data_vector[5], data_vector[6])

    # Run Ipopt with random init
    random_samples = np.random.rand(model.shots, model.dimension)
    opt_samples, current_best, _, solver_time = QHD.classicly_optimize(model.f_eval, random_samples,
                                                                       model.dimension,
                                                                       solver="IPOPT")
    ipopt_success_prob = calc_success_prob(current_best_qhd, opt_samples, model.shots, model.f_eval,
                                           tol)
    data_vector[8] = solver_time
    data_vector[9] = ipopt_success_prob

    # Run TNC with random init
    opt_samples, current_best, _, solver_time = QHD.classicly_optimize(model.f_eval, random_samples,
                                                                       model.dimension,
                                                                       solver="TNC")
    _, _, solver_time = model.post_process()
    tnc_success_prob = calc_success_prob(current_best_qhd, opt_samples, model.shots, model.f_eval,
                                         tol)

    data_vector[10] = solver_time
    data_vector[11] = tnc_success_prob

    return data_vector
