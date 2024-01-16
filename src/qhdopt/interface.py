import random
import time

import cyipopt
import jax.numpy as jnp
import numpy as np
import qutip as qtp
from jax import grad, jacfwd, jacrev, jit
from scipy.optimize import Bounds, minimize
from simuq import QSystem, Qubit, hlist_sum
from simuq.dwave.dwave_provider import DWaveProvider
from simuq.ionq import IonQAPICircuit, IonQProvider
from simuq.qutip import QuTiPProvider
from sympy import N, expand, lambdify, symbols


class QHD:
    def __init__(
        self,
        func,
        syms,
        bounds=None,
        custom_ordering=lambda sym: str(sym),
    ):
        self.raw_result = None
        self.lambda_numpy = lambdify(syms, func, jnp)
        self.list_of_symbols = syms
        self.dimension = len(func.free_symbols)
        self.qhd_samples = None
        self.post_processed_samples = None
        self.custom_ordering = custom_ordering
        self.info = dict()

        if bounds is None:
            self.lb = [0] * self.dimension
            self.scaling_factor = [1] * self.dimension
        elif isinstance(bounds, tuple):
            self.lb = [bounds[0]] * self.dimension
            self.scaling_factor = [bounds[1] - bounds[0]] * self.dimension
        elif isinstance(bounds, list):
            self.lb = [bounds[i][0] for i in range(self.dimension)]
            self.scaling_factor = [bounds[i][1] - bounds[i][0]] * self.dimension
        else:
            raise Exception(
                "Unsupported bounds type. Try: (lb, ub) or [(lb1, ub1), ..., (lbn, ubn)]."
            )
        self.univariate_dict, self.bivariate_dict = self.decompose_function(func)

    @classmethod
    def SymPy(cls, func, syms, bounds=None, custom_ordering=lambda sym: str(sym)):
        return cls(func, syms, bounds, custom_ordering)

    @classmethod
    def QP(cls, Q, b, bounds=None):
        f, xl = cls.quad_to_gen(Q, b)
        return cls(f, xl, bounds)

    @classmethod
    def quad_to_gen(cls, Q, b):
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
        anneal_schedule=[[0, 0], [20, 1]],
        penalty_coefficient=0,
        chain_strength=5e-2,
        penalty_ratio=0.75,
        post_processing_method="TNC",
    ):
        self.backend = "dwave"
        self.r, self.resolution = resolution, resolution
        self.shots = shots
        self.api_key = api_key
        if api_key_from_file != None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()
        self.embedding_scheme = embedding_scheme
        self.anneal_schedule = anneal_schedule
        self.penalty_coefficient = penalty_coefficient
        self.chain_strength = chain_strength
        self.penalty_ratio = penalty_ratio
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
        self.backend = "ionq"
        self.r, self.resolution = resolution, resolution
        self.shots = shots
        self.api_key = api_key
        if api_key_from_file != None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()
        self.embedding_scheme = embedding_scheme
        self.penalty_coefficient = penalty_coefficient
        self.discre, self.discretization = time_discretization, time_discretization
        self.gamma = gamma
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
        self.backend = "qutip"
        self.r, self.resolution = resolution, resolution
        self.shots = shots
        self.embedding_scheme = embedding_scheme
        self.penalty_coefficient = penalty_coefficient
        self.discre, self.discretization = time_discretization, time_discretization
        self.gamma = gamma
        self.post_processing_method = post_processing_method

    # Function to decompose a given function into univariate, bivariate, and trivariate parts
    def decompose_function(self, func):
        # Expand the function to simplify the decomposition
        affine_transformation_vars = list(
            self.affine_transformation(np.array(self.list_of_symbols))
        )
        func = func.subs(zip(self.list_of_symbols, affine_transformation_vars))
        lambdify_numpy = lambda free_vars, f: lambdify(free_vars, f, "numpy")
        symbol_to_int = {
            self.list_of_symbols[i]: i + 1 for i in range(len(self.list_of_symbols))
        }
        func_expanded = expand(func)

        # Containers for different parts
        univariate_terms, bivariate_terms = {}, {}

        # Iterate over the terms in the expanded form
        for term in func_expanded.as_ordered_terms():
            # Check the variables in the term
            vars_in_term = term.free_symbols

            # Classify the term based on the number of variables it contains
            if len(vars_in_term) == 1:
                single_var_index = symbol_to_int[list(vars_in_term)[0]]
                univariate_terms.setdefault(single_var_index, []).append(term)
            elif len(vars_in_term) == 2:
                index1, index2 = sorted(
                    [symbol_to_int[sym] for sym in list(vars_in_term)]
                )

                factors = term.as_ordered_factors()
                coefficient = 1
                i = 0
                while len(factors[i].free_symbols) == 0:
                    coefficient *= float(N(factors[i]))
                    i += 1

                f1, f2 = sorted(
                    [factors[i] for i in range(i, len(factors))],
                    key=lambda factor: symbol_to_int[list(factor.free_symbols)[0]],
                )
                bivariate_terms.setdefault((index1, index2), []).append(
                    (
                        coefficient,
                        lambdify_numpy(list(f1.free_symbols), f1),
                        lambdify_numpy(list(f2.free_symbols), f2),
                    )
                )
            elif len(vars_in_term) > 2:
                raise Exception(
                    f"The specified function has {len(vars_in_term)} variable term "
                    f"which is currently unsupported by QHD."
                )

        # Combine the terms to form each part
        univariate_part = {
            var: (1, lambdify_numpy(list(terms[0].free_symbols), sum(terms)))
            for var, terms in univariate_terms.items()
        }

        return univariate_part, bivariate_terms

    def affine_transformation(self, x):
        return self.scaling_factor * x + self.lb

    def f_eval(self, x):
        x = x.astype(jnp.float32)
        return self.lambda_numpy(*x)

    def unary_penalty(self, k, qubits):
        unary_penalty_sum = lambda k: sum(
            [
                qubits[j].Z * qubits[j + 1].Z
                for j in range(k * self.r, (k + 1) * self.r - 1)
            ]
        )
        return (
            (-1) * qubits[k * self.r].Z
            + qubits[(k + 1) * self.r - 1].Z
            - unary_penalty_sum(k)
        )

    def H_pen(self, qubits=None):
        if qubits is None:
            qubits = self.qubits
        if self.embedding_scheme == "hamming":
            return 0
        elif self.embedding_scheme == "unary":
            return hlist_sum(
                [self.unary_penalty(p, qubits) for p in range(self.dimension)]
            )

    def S_x(self, qubits=None):
        if qubits is None:
            qubits = self.qubits
        return hlist_sum([qubit.X for qubit in qubits])

    def H_k(self, qubits=None):
        if qubits is None:
            qubits = self.qubits
        if self.embedding_scheme == "onehot":

            def onehot_driving_sum(k):
                return sum(
                    [
                        0.5
                        * (
                            qubits[j].X * qubits[j + 1].X
                            + qubits[j].Y * qubits[j + 1].Y
                        )
                        for j in range(k * self.r, (k + 1) * self.r - 1)
                    ]
                )

            return (-0.5 * self.r**2) * hlist_sum(
                [onehot_driving_sum(p) for p in range(self.dimension)]
            )
        else:
            return (-0.5 * self.r**2) * hlist_sum([qubit.X for qubit in qubits])

    def H_p(self, qubits=None):
        if qubits is None:
            qubits = self.qubits

        # Encoding of the X operator as defined in (F.16) in https://browse.arxiv.org/pdf/2303.01471.pdf
        def Enc_X(k):
            S_z = lambda k: sum(
                [qubits[j].Z for j in range(k * self.r, (k + 1) * self.r)]
            )
            return (1 / 2) + (-1 / (2 * self.r)) * S_z(k)

        def get_ham(d, lmda):
            def n_j(d, j):
                return 0.5 * (
                    qubits[(d - 1) * self.r + j].I - qubits[(d - 1) * self.r + j].Z
                )

            if self.embedding_scheme == "unary":

                def eval_lmda_unary():
                    eval_points = [i / self.r for i in range(self.r + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_unary()
                H = eval_lmda[0] * self.qubits[(d - 1) * self.r].I
                for i in range(len(eval_lmda) - 1):
                    H += (eval_lmda[i + 1] - eval_lmda[i]) * n_j(d, self.r - i - 1)

                return H

            elif self.embedding_scheme == "onehot":

                def eval_lmda_onehot():
                    eval_points = [i / self.r for i in range(1, self.r + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_onehot()
                H = 0
                for i in range(len(eval_lmda)):
                    H += eval_lmda[i] * n_j(d, self.r - i - 1)

                return H

        H = 0
        for key, value in self.univariate_dict.items():
            coefficient, lmda = value
            if self.embedding_scheme == "hamming":
                H += lmda(Enc_X(key - 1))
            else:
                ham = get_ham(key, lmda)
                H += coefficient * ham

        for key, value in self.bivariate_dict.items():
            d1, d2 = key
            for term in value:
                coefficient, lmda1, lmda2 = term
                if self.embedding_scheme == "hamming":
                    H += coefficient * lmda1(Enc_X(d1 - 1)) * lmda2(Enc_X(d2 - 1))
                else:
                    H += coefficient * (get_ham(d1, lmda1) * get_ham(d2, lmda2))

        return H

    def calc_penalty_coefficient_and_chain_strength(self):
        qs = QSystem()
        qubits = [Qubit(qs) for _ in range(self.dimension * self.r)]
        qs.add_evolution(self.S_x(qubits) + self.H_p(qubits), 1)
        dwp = DWaveProvider(self.api_key)
        h, J = dwp.compile(qs, self.anneal_schedule)
        max_strength = np.max(np.abs(list(h) + list(J.values())))
        penalty_coefficient = (
            self.penalty_ratio * max_strength if self.embedding_scheme == "unary" else 0
        )
        chain_strength = np.max([5e-2, 0.5 * self.penalty_ratio])
        return penalty_coefficient, chain_strength

    def dwave_exec(self, verbose=0):
        (
            penalty_coefficient,
            chain_strength,
        ) = self.calc_penalty_coefficient_and_chain_strength()
        self.qs.add_evolution(
            self.S_x() + self.H_p() + penalty_coefficient * self.H_pen(), 1
        )

        dwp = DWaveProvider(self.api_key)
        self.prvd = dwp

        self.info["time_start_compile"] = time.time()
        dwp.compile(self.qs, self.anneal_schedule, chain_strength, self.shots)
        self.info["time_end_compile"] = time.time()
        if verbose > 1:
            print("Submit Task to D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        dwp.run(shots=self.shots)
        self.info["time_end_backend"] = time.time()
        self.info["average_qpu_time"] = dwp.avg_qpu_time
        if verbose > 1:
            print("Received Task from D-Wave:")
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

        self.raw_result = dwp.results()
        raw_samples = []
        for i in range(self.shots):
            raw_samples.append(QHD.spin_to_bitstring(self.raw_result[i]))

        return raw_samples

    def ionq_state_prep_one_hot(self, circ, amplitudes):
        def state_prep_one_hot_aux(n, starting_index, amplitudes):
            assert np.all(np.isreal(amplitudes)) and np.all(amplitudes >= 0)
            assert len(amplitudes) == n

            if n > 1:
                amplitudes_left = amplitudes[: int(n / 2)]
                amplitudes_right = amplitudes[int(n / 2) :]
                a = np.linalg.norm(amplitudes_left)

                if np.linalg.norm(amplitudes_right) > 0:
                    circ.rPP(
                        "X",
                        "Y",
                        starting_index,
                        starting_index + int(n / 2),
                        np.arccos(a),
                    )
                    circ.rPP(
                        "X",
                        "Y",
                        starting_index + int(n / 2),
                        starting_index,
                        -np.arccos(a),
                    )

                if np.linalg.norm(amplitudes_left) > 0:
                    state_prep_one_hot_aux(
                        int(n / 2),
                        starting_index,
                        amplitudes_left / np.linalg.norm(amplitudes_left),
                    )
                if np.linalg.norm(amplitudes_right) > 0:
                    state_prep_one_hot_aux(
                        int((n + 1) / 2),
                        starting_index + int(n / 2),
                        amplitudes_right / np.linalg.norm(amplitudes_right),
                    )

        n = self.r
        amplitudes = np.array(amplitudes)
        for i in range(self.dimension):
            amplitudes_abs_val = np.abs(amplitudes)

            # Start from 000...001 (first qubit on the right)
            circ.rx(i * n, np.pi)

            state_prep_one_hot_aux(n, i * n, amplitudes_abs_val)

            if not np.all(amplitudes >= 0):
                for k in range(self.r):
                    theta = np.angle(amplitudes[k])
                    circ.rz(i * n + k, theta)

    @staticmethod
    def binstr_to_bitstr(s):
        return list(map(int, list(s)))

    def ionq_exec(self, on_simulator=True, with_noise=False, verbose=0):
        if self.embedding_scheme != "onehot":
            raise Exception("IonQ backend must use one-hot embedding.")

        self.qs.add_evolution(10 * self.H_k(), 1)
        phi1 = lambda t: self.gamma / (1 + self.gamma * (t**2))
        phi2 = lambda t: self.gamma * (1 + self.gamma * (t**2))
        Ht = lambda t: phi1(t) * self.H_k() + phi2(t) * self.H_p()
        self.qs.add_td_evolution(Ht, np.linspace(0, 1, self.discre))

        iqp = IonQProvider(self.api_key)
        self.prvd = iqp

        num_sites = self.qs.num_sites
        state_prep = IonQAPICircuit(num_sites)
        self.ionq_state_prep_one_hot(
            state_prep, np.array([1] * self.r) / np.sqrt(self.r)
        )

        self.info["time_start_compile"] = time.time()
        iqp.compile(
            self.qs,
            backend="aria-1",
            trotter_num=1,
            state_prep=state_prep,
            verbose=-1,
            tol=0.1,
        )
        self.info["time_end_compile"] = time.time()

        iqp.run(shots=self.shots, on_simulator=on_simulator, with_noise=with_noise)
        self.raw_result = iqp.results(wait=1)
        raw_samples = []
        for k in self.raw_result:
            occ = int(self.raw_result[k] * self.shots)
            raw_samples += [k] * occ
        raw_samples = list(map(self.binstr_to_bitstr, raw_samples))
        self.info["time_end_backend"] = time.time()

        return raw_samples

    def qutip_exec(self, nsteps=10000, verbose=0):
        if self.embedding_scheme != "onehot":
            raise Exception("QuTiP simulation must use one-hot embedding.")

        self.qs.add_evolution(10 * self.H_k(), 1)
        phi1 = lambda t: self.gamma / (1 + self.gamma * (t**2))
        phi2 = lambda t: self.gamma * (1 + self.gamma * (t**2))
        Ht = lambda t: phi1(t) * self.H_k() + phi2(t) * self.H_p()
        self.qs.add_td_evolution(Ht, np.linspace(0, 1, self.discre))

        qpp = QuTiPProvider()
        self.prvd = qpp

        g = qtp.Qobj([[1], [0]])
        e = qtp.Qobj([[0], [1]])
        initial_state_per_dim = qtp.tensor([e] + [g] * (self.r - 1))
        for j in range(self.r - 1):
            initial_state_per_dim += qtp.tensor(
                [g] * (j + 1) + [e] + [g] * (self.r - 2 - j)
            )
        initial_state_per_dim = initial_state_per_dim / np.sqrt(self.r)

        initial_state = qtp.tensor([initial_state_per_dim] * self.dimension)

        self.info["time_start_compile"] = time.time()

        qpp.compile(self.qs, initial_state=initial_state)
        self.info["time_end_compile"] = time.time()

        qpp.run(nsteps=nsteps)
        self.raw_result = qpp.results()
        raw_samples = random.choices(
            list(self.raw_result.keys()), weights=self.raw_result.values(), k=self.shots
        )
        raw_samples = list(map(self.binstr_to_bitstr, raw_samples))
        self.info["time_end_backend"] = time.time()

        return raw_samples

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

    @staticmethod
    def hamming_bitstring_to_vec(bitstring, d, r):
        sample = np.zeros(d)
        for i in range(d):
            sample[i] = sum(bitstring[i * r : (i + 1) * r]) / r
        return sample

    @staticmethod
    def unary_bitstring_to_vec(bitstring, d, r):
        sample = np.zeros(d)

        for i in range(d):
            x_i = bitstring[i * r : (i + 1) * r]

            in_low_energy_subspace = True
            for j in range(r - 1):
                if x_i[j] > x_i[j + 1]:
                    in_low_energy_subspace = False

            if in_low_energy_subspace:
                sample[i] = np.mean(x_i)
            else:
                return None

        return sample

    @staticmethod
    def onehot_bitstring_to_vec(bitstring, d, r):
        sample = np.zeros(d)

        for i in range(d):
            x_i = bitstring[i * r : (i + 1) * r]
            if sum(x_i) != 1:
                return None
            else:
                slot = 0
                while slot < r and x_i[slot] == 0:
                    slot += 1
                sample[i] = 1 - slot / r

        return sample

    def bitstring_to_vec(self, bitstring, d, r):
        if self.embedding_scheme == "unary":
            return QHD.unary_bitstring_to_vec(bitstring, d, r)
        elif self.embedding_scheme == "onehot":
            return QHD.onehot_bitstring_to_vec(bitstring, d, r)
        elif self.embedding_scheme == "hamming":
            return QHD.hamming_bitstring_to_vec(bitstring, d, r)
        else:
            raise Exception("Illegal embedding scheme.")

    def decoder(self, raw_samples):
        qhd_samples = []
        minimizer = np.zeros(self.dimension)
        minimum = float("inf")

        for i in range(len(raw_samples)):
            bitstring = raw_samples[i]
            qhd_samples.append(self.bitstring_to_vec(bitstring, self.dimension, self.r))
            if qhd_samples[i] is None:
                continue
            if self.f_eval(qhd_samples[i]) < minimum:
                minimum = self.f_eval(qhd_samples[i])
                minimizer = qhd_samples[i]

        self.qhd_samples = qhd_samples

        return minimizer, minimum

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
        self.qs = QSystem()
        self.qubits = [Qubit(self.qs) for _ in range(self.dimension * self.r)]

        if self.backend == "dwave":
            raw_samples = self.dwave_exec(verbose=verbose)
        elif self.backend == "ionq":
            raw_samples = self.ionq_exec(verbose=verbose, with_noise=True)
        elif self.backend == "qutip":
            raw_samples = self.qutip_exec(verbose=verbose)

        self.raw_samples = raw_samples

        coarse_minimizer, coarse_minimum = self.decoder(raw_samples)
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
