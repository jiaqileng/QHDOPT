from qhdopt.utils.benchmark_utils import calc_success_prob


class Response:
    def __init__(self, coarse_samples, coarse_minimum, coarse_minimizer, affine_trans, info,
                 refined_samples=None, refined_minimum=None, refined_minimizer=None):
        self.unit_box_coarse_samples = coarse_samples
        self.coarse_minimum = coarse_minimum
        self.unit_box_coarse_minimizer = coarse_minimizer
        self.affine_trans = affine_trans
        self.info = info
        self.unit_box_refined_samples = refined_samples
        self.unit_box_refined_minimizer = refined_minimizer
        self.refined_minimum = refined_minimum
        self.coarse_minimizer = self.affine_trans(self.unit_box_coarse_minimizer)

        self.minimum = self.refined_minimum if refined_minimum is not None else self.coarse_minimum
        self.unit_box_minimizer = self.unit_box_refined_minimizer if self.unit_box_refined_minimizer is not None else self.unit_box_coarse_minimizer
    @property
    def coarse_samples(self):
        return [self.affine_trans(sample) for sample in self.unit_box_coarse_samples]

    @property
    def samples(self):
        samples = self.unit_box_refined_samples if self.unit_box_refined_samples is not None else self.unit_box_coarse_samples
        return [self.affine_trans(sample) for sample in samples]

    @property
    def minimizer(self):
        minimizer = self.unit_box_refined_minimizer if self.unit_box_refined_minimizer is not None else self.unit_box_coarse_minimizer
        return self.affine_trans(minimizer)

    def print_solver_info(self):
        print("* Coarse solution")
        print("Unit Box Minimizer:", self.unit_box_coarse_minimizer)
        print("Minimizer:", self.coarse_minimizer)
        print("Minimum:", self.coarse_minimum)
        print()

        if self.unit_box_refined_samples is not None:
            print("* Fine-tuned solution")
            print("Minimizer:", self.minimizer)
            print("Minimum:", self.minimum)
            print("Unit Box Minimizer:", self.unit_box_refined_minimizer)
            print()

    def print_time_info(self):

        total_runtime = self.info["compile_time"] + self.info["backend_time"] + self.info["decoding_time"]

        print("* Runtime breakdown")
        print(f"SimuQ compilation: {self.info['compile_time']:.3f} s")
        print(f"Backend runtime: {self.info['backend_time']:.3f} s")
        print(f"Decoding time: {self.info['decoding_time']:.3f} s")
        if self.info["fine_tune_status"]:
            print(f"Fine-tuning time: {self.info['fine_tuning_time']:.3f} s")
            total_runtime += self.info['fine_tuning_time']

        print(f"* Total time: {total_runtime:.3f} s")
