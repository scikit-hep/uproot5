# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os.path

from setuptools import setup


def get_version():
    g = {}
    with open(os.path.join("src", "uproot", "version.py")) as f:
        exec(f.read(), g)
    return g["__version__"]


extras = {
    "dev": [
        "awkward>=1.9.0rc1",
        "pandas",
        "boost_histogram>=0.13",
        "hist>=1.2",
        'dask[array];python_version >= "3.8"',
        'dask-awkward;python_version >= "3.8"',
    ],
    "test": [
        "awkward>=1.9.0rc1",
        "pytest>=4.6",
        "flake8",
        "flake8-print>=5",
        "scikit-hep-testdata",
        "lz4",
        "xxhash",
        "requests",
    ],
}
extras["all"] = sum(extras.values(), [])

setup(version=get_version(), extras_require=extras)
