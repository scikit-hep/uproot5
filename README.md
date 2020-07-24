<img src="docs-img/logo/logo-300px.png">

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Uproot is a reader and a writer of the [ROOT file format](https://root.cern/) using only Python and Numpy. Unlike the standard C++ ROOT implementation, Uproot is only an I/O library, primarily intended to stream data into machine learning libraries in Python. Unlike PyROOT and root_numpy, uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.

<p align="center"><img src="docs-img/diagrams/abstraction-layers.png" width="700px"></p>

# Uproot 3 vs Uproot 4

We are in the middle of a transition in which [Uproot 3](https://github.com/scikit-hep/uproot#readme) is deprecated in favor of Uproot 4. Most features are available in Uproot 4 (the major exception being file-writing) and it's ready for new physics analyses. Interfaces differ slightly between Uproot 3 and 4, and the new one isn't properly documented yet.

This [tutorial at PyHEP 2020](https://youtu.be/ea-zYLQBS4U) (video with [interactive notebook on Binder](https://mybinder.org/v2/gh/jpivarski/2020-07-13-pyhep2020-tutorial.git/1.1?urlpath=lab/tree/tutorial.ipynb)) may be a good way to get started, though it's understandable if you want to wait for full documentation.

Both libraries can be used in the same Python process; just

```bash
pip install uproot    # old
pip install uproot4   # new
```

and import them as `uproot` and `uproot4`, respectively. Later this year (2020), the two packages will shift to

```bash
pip install uproot    # new
pip install uproot3   # old
```

Note that Uproot 3 returns old-style [Awkward 0](https://github.com/scikit-hep/awkward-array#readme) arrays and Uproot 4 returns new-style [Awkward 1](https://github.com/scikit-hep/awkward-1.0#readme) arrays. (The new version of Uproot was motivated by the new version of Awkward, to make a clear distinction.)

<p align="center"><img src="docs-img/photos/switcheroo.jpg" width="400px"></p>

# Installation

Install uproot like any other Python package:

```bash
pip install uproot     # maybe with sudo or --user, -U to update, or in venv
```

# Dependencies

**Uproot 4's only strict dependency is NumPy.** (The pip command above will install it, if you don't have it.)

If you use any features that require more dependencies, you will be prompted with instructions to install them.

The full list is

   * `awkward1`: highly recommended, but can be avoided by passing `library="np"` to any functions that read arrays.
   * `pandas`: only if `library="pd"`.
   * `cupy`: only if `library="cp"` (reads arrays onto GPUs).
   * `dask[array]` and `dask[dataframe]`: experimental, for lazy arrays with `library="da"`.
   * `xrootd`: only if reading files with `root://` URLs.
   * `lz4` and `xxhash`: only if reading ROOT files that have been LZ4-compressed.
   * `zstandard`: only if reading ROOT files that have been ZSTD-compressed.
   * `backports.lzma`: only if reading ROOT files that have been LZMA-compressed (in Python 2).
   * `boost-histogram`: only if converting histograms to Boost with `.to_boost()`.
   * `hist`: only if converting histograms to hist with `.to_hist()`.
