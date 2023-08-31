# Non-negative similarity matching in PyTorch

[![PyPI Version](https://img.shields.io/pypi/v/pynsm.svg)](https://pypi.org/project/pynsm/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/downloads/release/python-380/)
[![License](https://img.shields.io/pypi/l/pynsm.svg)](https://github.com/Shagesh/pytorch-NSM/blob/master/LICENSE)

This is an implementation of non-negative similarity matching (NSM) for PyTorch focusing on ease of use, extensibility, and speed.

## Getting started

You can install the package from PyPI by using

```sh
pip install pynsm
```

## User documentation
Find examples, how-to guides, tutorials, and full API reference information on Readthedocs, https://pynsm.readthedocs.io/.

## Questions or issues?

Please contact us by opening an issue on GitHub.

<br/>

*********************************************************

<br/>

## Instructions for developers

### Developer installation

It is strongly recommended to use a virtual environment when working with this code. The installation instructions below include the commands for creating the virtual environment, using either `conda` (recommended) or `venv`.

#### Developer install using `conda`

If you do not have `conda` installed, the easiest way to get started is with [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Follow the installation instructions for your system.

Next, create a new environment and install for CPU using

```sh
conda env create -f environment.yml
```

For using an NVIDIA GPU run

```sh
conda env create -f environment-cuda.yml
```

Note that most Macs do not have an NVIDIA GPU, so you should use the first invocation shown above. If your Mac uses the newer Apple chips, you may be able to use ``device=mps`` to get GPU acceleration (the installation procedure remains unchanged).

The commands above automatically perform an "editable" install—this means that changes made to the code will automatically take effect without having to reinstall the package.

#### Developer install using `venv`

Before creating a new virtual environment, it is best to ensure you're not using the system version of Python—this is often badly out of date. Some options for doing this are outlined in [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/#installation-guides), although many options exist. One advantage of using `conda` is that this is done for you.

Once you have a proper Python install, create a new virtual environment by running the following command in a terminal inside the main folder of the repository:

```sh
python -m venv env
```

This creates a subfolder called `env` containing the files for the virtual environment. Next we need to activate the environment and install the package with its pre-requisites:

```sh
source env/bin/activate
pip install -e ".[dev]"
```

The `-e` marks this as an "editable" install—this means that changes made to the code will automatically take effect without having to reinstall the package.

### Example usage

See the notebooks in the [`examples`](examples) folder to get started with the package. The information on [readthedocs](https://pynsm.readthedocs.io/) may also prove useful.
