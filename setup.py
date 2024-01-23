from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("src/qhdopt/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
    name="qhdopt",
    version=version,
    description="A software package for nonconvex optimization with quantum devices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "jax[cpu]",
        "sympy",
        "scipy==1.10.1",
        "numpy",
        "jaxlib"
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
