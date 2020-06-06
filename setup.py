# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

import os.path
from setuptools import setup
from setuptools import find_packages

def get_version():
    g = {}
    exec(open(os.path.join("uproot4", "version.py")).read(), g)
    return g["__version__"]

setup(name = "uproot4",
      packages = find_packages(exclude = ["tests"]),
      scripts = [],
      version = get_version(),
      author = "Jim Pivarski",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski",
      maintainer_email = "pivarski@princeton.edu",
      description = "Development of uproot 4.0, to replace scikit-hep/uproot in 2020.",
      long_description = """(FIXME)""",
      long_description_content_type = "text/markdown",
      url = "https://github.com/scikit-hep/uproot4",
      download_url = "https://github.com/scikit-hep/uproot4/releases",
      license = "BSD 3-clause",
      test_suite = "tests",
      python_requires = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*",
      install_requires = ["numpy"],
      tests_require = ["pytest", "flake8", "scikit-hep-testdata", "pandas", "awkward1"],

      classifiers = [
#         "Development Status :: 1 - Planning",
#         "Development Status :: 2 - Pre-Alpha",
          "Development Status :: 3 - Alpha",
#         "Development Status :: 4 - Beta",
#         "Development Status :: 5 - Production/Stable",
#         "Development Status :: 6 - Mature",
#         "Development Status :: 7 - Inactive",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.6",
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
