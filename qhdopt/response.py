class Response:
    def __init__(self, info,
                 coarse_samples=None, coarse_minimum=None, coarse_minimizer=None,
                 refined_samples=None, refined_minimum=None, refined_minimizer=None, func=None):
        self.coarse_samples = coarse_samples
        self.coarse_minimum = coarse_minimum
        self.coarse_minimizer = coarse_minimizer
        self.refined_samples = refined_samples
        self.refined_minimum = refined_minimum
        self.refined_minimizer = refined_minimizer
        self.samples = self.refined_samples if self.refined_samples is not None else self.coarse_samples
        self.minimizer = self.refined_minimizer if refined_minimizer is not None else self.coarse_minimizer
        self.minimum = self.refined_minimum if refined_minimum is not None else self.coarse_minimum
        self.info = info
        self.func = func

    def get_percentage_in_embedding_subspace(self):
        number_in_subspace = sum([0 if el is None else 1 for el in self.samples])
        return number_in_subspace / len(self.samples)

    def get_success_probability(self, tol=1e-3, minimum=None):
        if self.func == None:
            raise Exception("No function to evaluate the samples.")

        if minimum is None:
            minimum = self.minimum
        successes = 0
        for sample in self.samples:
            if sample is not None and abs(self.func(sample) - minimum) < tol:
                successes +=1
        return successes / len(self.samples)


    def print_solver_info(self):
        print("* Coarse solution")
        # print("Unit Box Minimizer:", self.unit_box_coarse_minimizer)
        print("Minimizer:", self.coarse_minimizer)
        print("Minimum:", self.coarse_minimum)
        print()

        if self.refined_samples is not None:
            print("* Fine-tuned solution")
            print("Minimizer:", self.minimizer)
            print("Minimum:", self.minimum)
            # print("Unit Box Minimizer:", self.unit_box_refined_minimizer)
            print()

    def print_time_info(self):

        total_runtime = self.info["compile_time"] + self.info["backend_time"] + self.info["decoding_time"]

        print("* Runtime breakdown")
        print(f"SimuQ compilation: {self.info['compile_time']:.3f} s")
        print(f"Backend runtime: {self.info['backend_time']:.3f} s")
        print(f"Decoding time: {self.info['decoding_time']:.3f} s")
        if self.info["refine_status"]:
            print(f"Classical (Fine-tuning) time: {self.info['refining_time']:.3f} s")
            total_runtime += self.info['refining_time']

        print(f"* Total time: {total_runtime:.3f} s")
        print()
