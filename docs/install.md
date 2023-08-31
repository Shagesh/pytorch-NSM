# Installation instructions

## Basic install

The package can be installed directly from PyPI. Make sure to activate a Python virtual environment (see [below](#creating-a-virtual-environment)), and run

```bash
pip install pynsm
```

## Development install

A development install is only needed if you want to extend the `pynsm` package itself. In this case, clone [the repository](https://github.com/Shagesh/pytorch-NSM), then follow the [developer installation instructions on GitHub](https://github.com/Shagesh/pytorch-NSM/blob/main/README.md#developer-installation).

## Creating a virtual environment

It is good practice to use different virtual environments for different projects, to avoid complications due to different package versions. There are multiple ways of doing this, but the two most common ones are using `conda` or `venv`.

### Using `conda`

The `conda` package manager can be used to install and manage Python on your system. The easiest way to get started is using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).[^1]

[^1]: We do not recommend using a comprehensive distribution like Anaconda, as this can make it very difficult to work on multiple projects with potentially different pre-requisites.

To create a new virtual environment, run

```bash
conda create --name pynsm_env python=3.8
```

You can of course substitute `pynsm_env` with the environment name of your choice. You can also use a different Python version, though note that the package might not work with versions below 3.8.

Before using the environment, it needs to be activated using

```bash
conda activate pynsm_env
```

(Again, substitute the name you chose for the environment.)

### Using `venv`

Before creating a virtual environment using this method, ensure that you have a proper Python install that does not rely on the system Python interpreter. The reason is that the system Python is often badly out of date and any upgrades to it or new package installs can lead to problems with components of your OS that rely on the original versions.

Some options for installing Python are outlined in [The Hitchhiker's Guide to Python](https://docs.python-guide.org/starting/installation/#installation-guides), although many options exist. One advantage of using `conda` is that this step is done automatically.

Once you have a proper Python install, you can create a new virtual environment by running

```bash
python -m venv env
```

Note that this creates a subfolder of the current folder called `env` containing the files for the virtual environment. The name of the environment folder can be changed by using a different name in the command above.

As with `conda`, the environment needs to be activated before use. This can be done by running

```bash
source env/bin/activate
```
