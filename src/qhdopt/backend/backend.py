from abc import ABC, abstractmethod

from simuq import QSystem, hlist_sum


class Backend(ABC):
    def __init__(self, resolution, shots, embedding_scheme, qs, qubits, univariate_dict, bivariate_dict):
        self.qs = qs
        self.qubits = qubits
        self.resolution = resolution
        self.dimension = len(qubits) / self.resolution  # dimension * resolution = len(qubits)
        self.shots = shots
        self.embedding_scheme = embedding_scheme
        self.univariate_dict = univariate_dict
        self.bivariate_dict = bivariate_dict

    def S_x(self, qubits):
        return hlist_sum([qubit.X for qubit in qubits])

    def unary_penalty(self, k, qubits):
        unary_penalty_sum = lambda k: sum(
            [
                qubits[j].Z * qubits[j + 1].Z
                for j in range(k * self.resolution, (k + 1) * self.resolution - 1)
            ]
        )
        return (
                (-1) * qubits[k * self.resolution].Z
                + qubits[(k + 1) * self.resolution - 1].Z
                - unary_penalty_sum(k)
        )

    def H_pen(self, qubits):
        if self.embedding_scheme == "hamming":
            return 0
        elif self.embedding_scheme == "unary":
            return hlist_sum(
                [self.unary_penalty(p, qubits) for p in range(self.dimension)]
            )

    def H_p(self, qubits, univariate_dict, bivariate_dict):

        # Encoding of the X operator as defined in (F.16) in https://browse.arxiv.org/pdf/2303.01471.pdf
        def Enc_X(k):
            S_z = lambda k: sum(
                [qubits[j].Z for j in range(k * self.resolution, (k + 1) * self.resolution)]
            )
            return (1 / 2) + (-1 / (2 * self.resolution)) * S_z(k)

        def get_ham(d, lmda):
            def n_j(d, j):
                return 0.5 * (
                        qubits[(d - 1) * self.resolution + j].I - qubits[
                    (d - 1) * self.resolution + j].Z
                )

            if self.embedding_scheme == "unary":

                def eval_lmda_unary():
                    eval_points = [i / self.resolution for i in range(self.resolution + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_unary()
                H = eval_lmda[0] * qubits[(d - 1) * self.resolution].I
                for i in range(len(eval_lmda) - 1):
                    H += (eval_lmda[i + 1] - eval_lmda[i]) * n_j(d, self.resolution - i - 1)

                return H

            elif self.embedding_scheme == "onehot":

                def eval_lmda_onehot():
                    eval_points = [i / self.resolution for i in range(1, self.resolution + 1)]
                    return [lmda(x) for x in eval_points]

                eval_lmda = eval_lmda_onehot()
                H = 0
                for i in range(len(eval_lmda)):
                    H += eval_lmda[i] * n_j(d, self.resolution - i - 1)

                return H

        H = 0
        for key, value in univariate_dict.items():
            coefficient, lmda = value
            if self.embedding_scheme == "hamming":
                H += lmda(Enc_X(key - 1))
            else:
                ham = get_ham(key, lmda)
                H += coefficient * ham

        for key, value in bivariate_dict.items():
            d1, d2 = key
            for term in value:
                coefficient, lmda1, lmda2 = term
                if self.embedding_scheme == "hamming":
                    H += coefficient * lmda1(Enc_X(d1 - 1)) * lmda2(Enc_X(d2 - 1))
                else:
                    H += coefficient * (get_ham(d1, lmda1) * get_ham(d2, lmda2))

        return H

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
                        for j in range(k * self.resolution, (k + 1) * self.resolution - 1)
                    ]
                )

            return (-0.5 * self.resolution ** 2) * hlist_sum(
                [onehot_driving_sum(p) for p in range(self.dimension)]
            )
        else:
            return (-0.5 * self.resolution ** 2) * hlist_sum([qubit.X for qubit in qubits])

    @abstractmethod
    def exec(self, verbose, info):
        pass
