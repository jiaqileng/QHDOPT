from qhdopt.backend import Backend


class DwaveBackend(Backend):
    def __init__(self,
                 resolution,
                 shots=100,
                 api_key=None,
                 api_key_from_file=None,
                 embedding_scheme="unary",
                 anneal_schedule=None,
                 penalty_coefficient=0,
                 chain_strength=None,
                 penalty_ratio=0.75, ):
        if anneal_schedule is None:
            anneal_schedule = [[0, 0], [20, 1]]
        self.backend = "dwave"
        self.r, self.resolution = resolution, resolution
        self.shots = shots
        self.api_key = api_key
        if api_key_from_file is not None:
            with open(api_key_from_file, "r") as f:
                self.api_key = f.readline().strip()
        self.embedding_scheme = embedding_scheme
        self.anneal_schedule = anneal_schedule
        self.penalty_coefficient = penalty_coefficient
        self.chain_strength = chain_strength
        self.penalty_ratio = penalty_ratio

    def exec(self, verbose):
        print("sdf")
