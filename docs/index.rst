.. QHDOPT documentation master file, created by
   sphinx-quickstart on Sun Feb 25 22:13:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QHDOPT
==================================
QHDOPT (QHD-based OPTimizer) is a software package for nonconvex optimization.

QHDOPT implements a quantum optimization algorithm named Quantum Hamiltonian Descent (QHD) on available quantum computers (such as the D-Wave systems).
QHD is a quantum-upgraded version of gradient descent (GD).
Unlike the classical GD, QHD demonstrates a significant advantage in solving nonconvex optimization problems.

QHDOPT is for everyone!
QHDOPT aims to eliminate the technical barrier of using QHD for the broader operations research (OR) community.
We do not assume users to have prior knowledge of quantum computing, while we allow expert users to specify advanced solver parameters for customized experience. Our target users include:

Professionals pursuing an off-the-shelf nonconvex optimization solver to tackle problems in operations research (e.g., power systems, supply chains, manufacturing, health care, etc.),
Researchers who hope to advance the theory and algorithms of optimization via quantum technologies,
Experts in quantum computation who want to experiment with hyperparameters and/or encodings in QHD to achieve even better practical performance.
Fast compilation empowered by SimuQ
QHDOPT has a built-in compiler powered by SimuQ, a framework for programming and compiling quantum Hamiltonian systems.

Automatic post-processing
QHDOPT automatically post-processes the results returned by the quantum machines. The post-processing includes decoding the raw measurement results and improving their precision (i.e., fine-tuning) via a classical local solver. Users may disable the fine-tuning if needed.


.. toctree::
   api
