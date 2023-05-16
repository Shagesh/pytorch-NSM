import pathlib
from setuptools import setup

README = pathlib.Path("README.md").read_text()

setup(
    name="pynsm",
    version="0.0.1",
    description="A PyTorch implementation of non-negative similarity matching",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Yanis Bahroun, Shagesh Sridharan, Tiberiu Tesileanu",
    author_email="shagesh1996@gmail.com, "
    "ybahroun@flatironinstitute.org, "
    "ttesileanu@gmail.com",
    license="MIT",
    url="https://github.com/Shagesh/pytorch-NSM",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
    ],
    packages=["pynsm"],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "matplotlib",
        "seaborn",
        "torch",
        "torchvision",
        "scikit-learn",
        "ipykernel",
        "ipywidgets",
        "jupyter",
    ],
    include_package_data=True,
)
