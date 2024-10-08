from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("qhdopt/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
    name="qhdopt",
    version=version,
    description="A software package for nonconvex optimization with quantum devices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "jax[cpu]",
        "sympy",
        "scipy>=1.10.1,<1.12",
        "numpy<1.28.0",
        "jaxlib",
        "qutip<5",
        "simuq>=0.3.1",
        "dwave-system",
        "qiskit",
        "matplotlib"
    ],
    extras_require={
        "dev": ["tox"],
        "all": ["qhdopt[dev]"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
