# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

import os.path
from setuptools import setup


def get_version():
    g = {}
    with open(os.path.join("uproot", "version.py")) as f:
        exec(f.read(), g)
    return g["__version__"]


extras = {
    "test": open("requirements-test.txt").read().strip().split("\n"),
    "dev":  open("requirements-dev.txt").read().strip().split("\n"),
}
extras["all"] = sum(extras.values(), [])

setup(
    version = get_version(),
    extras_require = extras,
)
