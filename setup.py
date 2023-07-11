# Inspired by:
# https://github.com/dfm/emcee/blob/main/setup.py
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

import pathlib
import re
from setuptools import setup

NAME = "pynsm"
README = pathlib.Path("README.md").read_text()
META = (pathlib.Path("src") / "pynsm" / "__init__.py").read_text()
INSTALL_REQUIRES = ["numpy", "torch", "torchvision", "scikit-learn"]
SETUP_REQUIRES = ["setuptools"]
EXTRAS_REQUIRE = {
    "test": ["pytest"],
    "dev": ["pytest", "matplotlib", "seaborn", "ipykernel", "tqdm"],
    "examples": ["matplotlib", "seaborn", "ipykernel", "tqdm"],
}
CLASSIFIERS = [
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Intended Audience :: Science/Research",
]


def find_meta(meta):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), META, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


setup(
    name=NAME,
    version=find_meta("version"),
    author=find_meta("author"),
    author_email=find_meta("email"),
    license=find_meta("license"),
    url=find_meta("uri"),  # XXX should be link to readthedocs
    project_urls={"Source": "https://github.com/Shagesh/pytorch-NSM"},
    description=find_meta("description"),
    long_description=README,
    long_description_content_type="text/markdown",
    packages=["pynsm"],
    package_dir={"": "src"},
    classifiers=CLASSIFIERS,
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
