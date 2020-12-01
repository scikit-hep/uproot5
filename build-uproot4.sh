#!/usr/bin/env bash
# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

set -e

rm -rf build dist
mkdir build
mkdir build/uproot4

cat > build/uproot4/__init__.py << EOF
# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

from uproot import *
EOF

cat > build/uproot4-setup.py << EOF

import setuptools
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("uproot", "version.py")).read(), g)
    return g["__version__"]

setup(name = "uproot4",
      packages = ["uproot4"],
      package_dir = {"": "build"},
      version = get_version(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "ROOT I/O in pure Python and NumPy.",
      long_description = open("README.md").read(),
      long_description_content_type = "text/markdown",
      url = "https://github.com/scikit-hep/uproot4",
      download_url = "https://github.com/scikit-hep/uproot4/releases",
      license = "BSD 3-clause",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
      install_requires = ["uproot>=4.0.0rc1"],
      classifiers = [
#         "Development Status :: 1 - Planning",
#         "Development Status :: 2 - Pre-Alpha",
#         "Development Status :: 3 - Alpha",
#         "Development Status :: 4 - Beta",
          "Development Status :: 5 - Production/Stable",
#         "Development Status :: 6 - Mature",
#         "Development Status :: 7 - Inactive",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ])
EOF
