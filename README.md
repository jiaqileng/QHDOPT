# QHDOPT

[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](https://unitary.fund)<br>
[![Python CI](https://github.com/jiaqileng/QHDOPT/actions/workflows/python-ci.yml/badge.svg)](https://github.com/jiaqileng/QHDOPT/actions/workflows/python-ci.yml)
[![codecov](https://codecov.io/gh/jiaqileng/QHDOPT/graph/badge.svg?token=Z1GMD2AD8R)](https://codecov.io/gh/jiaqileng/QHDOPT)
[![Docs](https://img.shields.io/badge/docs-website-blue.svg)](https://jiaqileng.github.io/QHDOPT/)
<br>
**QHDOPT** (QHD-based OPTimizer) is a software package for nonlinear optimization.

QHDOPT implements a quantum optimization algorithm named [Quantum Hamiltonian Descent](https://jiaqileng.github.io/quantum-hamiltonian-descent/) (QHD) on available quantum computers (such as the [D-Wave systems](https://www.dwavesys.com/)). QHD is a quantum-upgraded version of gradient descent (GD). Unlike the classical GD, QHD demonstrates a significant advantage in solving nonconvex and nonlinear optimization problems.

<p align="center">
<img src="img/workflow.png" alt="QHDOPT Workflow" width="600">
</p>

## Why QHDOPT?

#### QHDOPT is for everyone!
QHDOPT aims to eliminate the technical barrier of using QHD for the broader operations research (OR) community. We do not assume users to have prior knowledge of quantum computing, while we allow expert users to specify advanced solver parameters for customized experience. Our target users include:

- Professionals pursuing an *off-the-shelf* nonconvex optimization solver to tackle problems in operations research (e.g., power systems, supply chains, manufacturing, health care, etc.),
- Researchers who hope to advance the theory and algorithms of optimization via quantum technologies,
- Experts in quantum computation who want to experiment with hyperparameters and/or encodings in QHD to achieve even better practical performance.

#### Fast compilation empowered by SimuQ
QHDOPT has a built-in compiler powered by [SimuQ](https://github.com/PicksPeng/SimuQ), a framework for programming and compiling quantum Hamiltonian systems.

#### Automatic post-processing
QHDOPT automatically post-processes the results returned by the quantum machines. The post-processing includes decoding the raw measurement results and improving their precision (i.e., fine-tuning) via a classical local solver. Users may disable the fine-tuning if needed.

## Installation

QHDOPT has a dependency on Ipopt. You may install Ipopt in your conda environment by

```bash
conda install -c conda-forge cyipopt==1.3.0
```

To install QHDOPT, you can directly install with `pip` by

```bash
pip install qhdopt
```

If you prefer to install from sources, clone this repo and install by

```bash
git clone https://github.com/jiaqileng/QHDOPT.git
cd QHDOPT/
pip install ".[all]"
```

## Usage

Two example notebooks for a jump start are `examples/1_quadratic_programming.ipynb` and `examples/2_nonlinear_programming.ipynb`. The following illustrates the basic building blocks of QHDOPT and their functionalities briefly.

Import QHDOPT by running

```python
from qhdopt import QHD
```

You can create a problem instance by directly constructing the function via SymPy.

```python
from sympy import symbols, exp

x, y = symbols("x y")
f = y**1.5 - exp(4*x) * (y-0.75)
model = QHD.SymPy(f, [x, y], bounds=(0,1))
```

Then you need to setup the solver and the backend device (D-Wave in this example).

```python
model.dwave_setup(resolution=8, api_key="API_key")
```

Here `resolution` represents the resolution of the QHD algorithm, and `api_key` represents the API key of the D-Wave account obtained at [D-Wave Leap](https://cloud.dwavesys.com/leap/).

Now you can solve the target problem.

```python
minimum = model.optimize()
```

The minimal value of $f$ found by QHDOPT is then stored in `minimum`. To print more details in the process, you can run `model.optimize(verbose=1)`.

## Contact
Jiaqi Leng [jiaqil@terpmail.umd.edu](mailto:jiaqil@terpmail.umd.edu)

Yuxiang Peng [pickspeng@gmail.com](mailto:pickspeng@gmail.com)

## Contributors
Samuel Kushnir, Jiaqi Leng, Yuxiang Peng, Lei Fan, Xiaodi Wu

## Citation

If you use QHDOPT in your work, please cite our paper

```
@misc{kushnir2024qhdopt,
  author    = {Kushnir, Sam and Leng, Jiaqi and Peng, Yuxiang and Fan, Lei and Wu, Xiaodi},
  publisher = {{INFORMS Journal on Computing}},
  title     = {{QHDOPT}: A Software for Nonlinear Optimization with {Q}uantum {H}amiltonian {D}escent},
  year      = {2024},
  doi       = {10.1287/ijoc.2024.0587.cd},
  url       = {https://github.com/INFORMSJoC/2024.0587},
  note      = {Available for download at https://github.com/INFORMSJoC/2024.0587},
}

