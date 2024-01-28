import time

from simuq import hlist_sum
from simuq.ionq import IonQProvider, IonQAPICircuit

from qhdopt.backend.backend import Backend
import numpy as np

from qhdopt.utils.decoding_utils import binstr_to_bitstr


class IonQBackend(Backend):
    def __init__(self,
                 resolution,
                 dimension,
                 univariate_dict,
                 bivariate_dict,
                 shots=100,
                 api_key=None,
                 api_key_from_file=None,
                 embedding_scheme="onehot",
                 penalty_coefficient=0,
                 time_discretization=10,
                 gamma=5,
                 on_simulator=False,
                 with_noise=False,
                 ):
        super().__init__(resolution, dimension, shots, embedding_scheme, univariate_dict,
                         bivariate_dict)
        self.api_key = api_key
        if api_key_from_file is not None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()
        self.penalty_coefficient = penalty_coefficient
        self.time_discretization = time_discretization
        self.gamma = gamma
        self.on_simulator = on_simulator
        self.with_noise = with_noise

    def ionq_state_prep_one_hot(self, circ, amplitudes):
        def state_prep_one_hot_aux(n, starting_index, amplitudes):
            assert np.all(np.isreal(amplitudes)) and np.all(amplitudes >= 0)
            assert len(amplitudes) == n

            if n > 1:
                amplitudes_left = amplitudes[: int(n / 2)]
                amplitudes_right = amplitudes[int(n / 2):]
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

        n = self.resolution
        amplitudes = np.array(amplitudes)
        for i in range(self.dimension):
            amplitudes_abs_val = np.abs(amplitudes)

            # Start from 000...001 (first qubit on the right)
            circ.rx(i * n, np.pi)

            state_prep_one_hot_aux(n, i * n, amplitudes_abs_val)

            if not np.all(amplitudes >= 0):
                for k in range(self.resolution):
                    theta = np.angle(amplitudes[k])
                    circ.rz(i * n + k, theta)

    def exec(self, verbose, info):
        if self.embedding_scheme != "onehot":
            raise Exception("IonQ backend must use one-hot embedding.")

        self.qs.add_evolution(10 * self.H_k(), 1)
        phi1 = lambda t: self.gamma / (1 + self.gamma * (t ** 2))
        phi2 = lambda t: self.gamma * (1 + self.gamma * (t ** 2))
        Ht = lambda t: phi1(t) * self.H_k() + phi2(t) * self.H_p(self.qubits, self.univariate_dict, self.bivariate_dict)
        self.qs.add_td_evolution(Ht, np.linspace(0, 1, self.time_discretization))

        iqp = IonQProvider(self.api_key)
        self.prvd = iqp

        num_sites = self.qs.num_sites
        state_prep = IonQAPICircuit(num_sites)
        self.ionq_state_prep_one_hot(
            state_prep, np.array([1] * self.resolution) / np.sqrt(self.resolution)
        )

        info["time_start_compile"] = time.time()
        iqp.compile(
            self.qs,
            backend="aria-1",
            trotter_num=1,
            state_prep=state_prep,
            verbose=-1,
            tol=0.1,
        )
        info["time_end_compile"] = time.time()

        iqp.run(shots=self.shots, on_simulator=self.on_simulator, with_noise=self.with_noise)
        self.raw_result = iqp.results(wait=1)
        raw_samples = []
        for k in self.raw_result:
            occ = int(self.raw_result[k] * self.shots)
            raw_samples += [k] * occ
        raw_samples = list(map(binstr_to_bitstr, raw_samples))
        info["time_end_backend"] = time.time()

        return raw_samples
