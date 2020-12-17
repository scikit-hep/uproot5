<img src="https://raw.githubusercontent.com/scikit-hep/uproot4/main/docs-img/logo/logo-300px.png">

[![PyPI version](https://badge.fury.io/py/uproot.svg)](https://pypi.org/project/uproot)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uproot)](https://github.com/conda-forge/uproot-feedstock)
[![Python 3.5‒3.9](https://img.shields.io/badge/Python-3.5%E2%80%923.9-blue)](https://www.python.org)
[![BSD-3 Clause License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Test build](https://img.shields.io/github/workflow/status/scikit-hep/uproot4/Test%20build?label=CI)](https://github.com/scikit-hep/uproot4/actions?query=workflow%3A%22Test+build%22)
[![Deploy to PyPI](https://img.shields.io/github/workflow/status/scikit-hep/uproot4/Deploy%20to%20PyPI?label=CD)](https://github.com/scikit-hep/uproot4/actions?query=workflow%3A%22Deploy+to+PyPI%22)

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![DOI 10.5281/zenodo.3952728](https://zenodo.org/badge/DOI/10.5281/zenodo.3952728.svg)](https://doi.org/10.5281/zenodo.3952728)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://uproot.readthedocs.io/)
[![Gitter](https://img.shields.io/badge/chat-online-success)](https://gitter.im/Scikit-HEP/uproot)

Uproot is a reader and a writer of the [ROOT file format](https://root.cern/) using only Python and Numpy. Unlike the standard C++ ROOT implementation, Uproot is only an I/O library, primarily intended to stream data into machine learning libraries in Python. Unlike PyROOT and root_numpy, Uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.

<p align="center"><img src="https://raw.githubusercontent.com/scikit-hep/uproot4/main/docs-img/diagrams/abstraction-layers.png" width="700px"></p>

# Installation

Uproot can be installed [from PyPI](https://pypi.org/project/uproot) using pip ([Awkward Array](https://pypi.org/project/awkward) is optional but highly recommended):

```bash
pip install uproot awkward
```

Uproot is also available using [conda](https://anaconda.org/conda-forge/uproot) (so is [Awkward Array](https://anaconda.org/conda-forge/awkward), which conda installs automatically):

```bash
conda install -c conda-forge uproot
```

If you have already added `conda-forge` as a channel, the `-c conda-forge` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions:

```bash
conda config --add channels conda-forge
conda update --all
```

**Note:** if you need to _write_ ROOT files, you'll need to use the deprecated [uproot3](https://github.com/scikit-hep/uproot) for now. This feature is coming to the new version soon.

## Getting help

**Start with the [tutorials and reference documentation](https://uproot.readthedocs.io/).**

   * Report bugs, request features, and ask for additional documentation on [GitHub Issues](https://github.com/scikit-hep/uproot4/issues).
   * If you have a "How do I...?" question, start a [GitHub Discussion](https://github.com/scikit-hep/uproot4/discussions) with category "Q&A".
   * Alternatively, ask about it on [StackOverflow with the [uproot] tag](https://stackoverflow.com/questions/tagged/uproot). Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
   * To ask questions in real time, try the Gitter [Scikit-HEP/uproot](https://gitter.im/Scikit-HEP/uproot) chat room.

## Installation for developers

Uproot is an ordinary Python library; you can get a copy of the code with

```bash
git clone https://github.com/scikit-hep/uproot4.git
```

and install it locally by calling `pip install .` in the repository directory.

If you need to develop Awkward Array as well, see its [installation for developers](https://github.com/scikit-hep/awkward-1.0#installation-for-developers).

# Dependencies

**Uproot's only strict dependency is NumPy.** This is the only dependency that pip will automatically install.

**Awkward Array is highly recommended.** It is not a strict dependency to allow Uproot to be used in restrictive environments. If you're using Uproot without Awkward Array, you'll have to use the `library="np"` option or globally set `uproot.default_library` to return arrays as NumPy arrays (see documentation).

   * `awkward`: be sure to use Awkward Array 1.x.

The following libraries are also useful in conjunction with Uproot, but are not necessary. If you call a function that needs one, you'll be prompted to install it. (Conda installs most of these automatically.)

**For ROOT files, compressed different ways:**

   * `lz4` and `xxhash`: only if reading ROOT files that have been LZ4-compressed.
   * `zstandard`: only if reading ROOT files that have been ZSTD-compressed.
   * `backports.lzma`: only if reading ROOT files that have been LZMA-compressed (in Python 2).

**For remote data:**

   * `xrootd`: only if reading files with `root://` URLs.

**For exporting data to other libraries:**

   * `pandas`: only if `library="pd"`.
   * `cupy`: only if `library="cp"` (reads arrays onto GPUs).
   * `boost-histogram`: only if converting histograms to [boost-histogram](https://github.com/scikit-hep/boost-histogram) with `histogram.to_boost()`.
   * `hist`: only if converting histograms to [hist](https://github.com/scikit-hep/hist) with `histogram.to_hist()`.

# Acknowledgements

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of Uproot contributors (including the [original repository](https://github.com/scikit-hep/uproot)).

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/jpivarski"><img src="https://avatars0.githubusercontent.com/u/1852447?v=4" width="100px;" alt=""/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=jpivarski" title="Code">💻</a> <a href="https://github.com/scikit-hep/uproot4/commits?author=jpivarski" title="Documentation">📖</a> <a href="#infra-jpivarski" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-jpivarski" title="Maintenance">🚧</a></td>
    <td align="center"><a href="https://github.com/reikdas"><img src="https://avatars0.githubusercontent.com/u/11775615?v=4" width="100px;" alt=""/><br /><sub><b>Pratyush Das</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=reikdas" title="Code">💻</a> <a href="#infra-reikdas" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/chrisburr"><img src="https://avatars3.githubusercontent.com/u/5220533?v=4" width="100px;" alt=""/><br /><sub><b>Chris Burr</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=chrisburr" title="Code">💻</a> <a href="#infra-chrisburr" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="https://github.com/plexoos"><img src="https://avatars0.githubusercontent.com/u/5005079?v=4" width="100px;" alt=""/><br /><sub><b>Dmitri Smirnov</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=plexoos" title="Code">💻</a></td>
    <td align="center"><a href="http://www.matthewfeickert.com/"><img src="https://avatars3.githubusercontent.com/u/5142394?v=4" width="100px;" alt=""/><br /><sub><b>Matthew Feickert</b></sub></a><br /><a href="#infra-matthewfeickert" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="http://www.tamasgal.com"><img src="https://avatars1.githubusercontent.com/u/1730350?v=4" width="100px;" alt=""/><br /><sub><b>Tamas Gal</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=tamasgal" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/kreczko"><img src="https://avatars3.githubusercontent.com/u/1213276?v=4" width="100px;" alt=""/><br /><sub><b>Luke Kreczko</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=kreczko" title="Code">💻</a> <a href="https://github.com/scikit-hep/uproot4/commits?author=kreczko" title="Tests">⚠️</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/nsmith-"><img src="https://avatars2.githubusercontent.com/u/6587412?v=4" width="100px;" alt=""/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=nsmith-" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/nbiederbeck"><img src="https://avatars1.githubusercontent.com/u/15156697?v=4" width="100px;" alt=""/><br /><sub><b>Noah Biederbeck</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=nbiederbeck" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/oshadura"><img src="https://avatars2.githubusercontent.com/u/7012420?v=4" width="100px;" alt=""/><br /><sub><b>Oksana Shadura</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=oshadura" title="Code">💻</a> <a href="#infra-oshadura" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
    <td align="center"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4" width="100px;" alt=""/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=henryiii" title="Code">💻</a> <a href="#infra-henryiii" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/scikit-hep/uproot4/commits?author=henryiii" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/masonproffitt"><img src="https://avatars3.githubusercontent.com/u/32773304?v=4" width="100px;" alt=""/><br /><sub><b>Mason Proffitt</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=masonproffitt" title="Code">💻</a> <a href="https://github.com/scikit-hep/uproot4/commits?author=masonproffitt" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/jonas-rembser/"><img src="https://avatars2.githubusercontent.com/u/6578603?v=4" width="100px;" alt=""/><br /><sub><b>Jonas Rembser</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=guitargeek" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/benkrikler"><img src="https://avatars0.githubusercontent.com/u/4083697?v=4" width="100px;" alt=""/><br /><sub><b>benkrikler</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=benkrikler" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/HDembinski"><img src="https://avatars0.githubusercontent.com/u/2631586?v=4" width="100px;" alt=""/><br /><sub><b>Hans Dembinski</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=HDembinski" title="Documentation">📖</a></td>
    <td align="center"><a href="http://marcelrieger.com"><img src="https://avatars0.githubusercontent.com/u/1908734?v=4" width="100px;" alt=""/><br /><sub><b>Marcel R.</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=riga" title="Code">💻</a></td>
    <td align="center"><a href="http://turra.web.cern.ch/turra/"><img src="https://avatars3.githubusercontent.com/u/143389?v=4" width="100px;" alt=""/><br /><sub><b>Ruggero Turra</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=wiso" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/jrueb"><img src="https://avatars2.githubusercontent.com/u/30041073?v=4" width="100px;" alt=""/><br /><sub><b>Jonas Rübenach</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=jrueb" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/bfis"><img src="https://avatars0.githubusercontent.com/u/15651150?v=4" width="100px;" alt=""/><br /><sub><b>bfis</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=bfis" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/raymondEhlers"><img src="https://avatars0.githubusercontent.com/u/1571927?v=4" width="100px;" alt=""/><br /><sub><b>Raymond Ehlers</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=raymondEhlers" title="Code">💻</a></td>
    <td align="center"><a href="http://andrzejnovak.github.io/"><img src="https://avatars1.githubusercontent.com/u/13226500?v=4" width="100px;" alt=""/><br /><sub><b>Andrzej Novak</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=andrzejnovak" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/bendavid"><img src="https://avatars2.githubusercontent.com/u/4920798?v=4" width="100px;" alt=""/><br /><sub><b>Josh Bendavid</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=bendavid" title="Code">💻</a></td>
    <td align="center"><a href="https://ddavis.io/"><img src="https://avatars2.githubusercontent.com/u/3202090?v=4" width="100px;" alt=""/><br /><sub><b>Doug Davis</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=douglasdavis" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/asymmetry"><img src="https://avatars3.githubusercontent.com/u/679529?v=4" width="100px;" alt=""/><br /><sub><b>Chao Gu</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=asymmetry" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/ast0815"><img src="https://avatars2.githubusercontent.com/u/5884065?v=4" width="100px;" alt=""/><br /><sub><b>Lukas Koch</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=ast0815" title="Code">💻</a></td>
    <td align="center"><a href="http://irfu.cea.fr/dap/"><img src="https://avatars1.githubusercontent.com/u/17836610?v=4" width="100px;" alt=""/><br /><sub><b>Michele Peresano</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=HealthyPear" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/EdoPro98"><img src="https://avatars1.githubusercontent.com/u/57357892?v=4" width="100px;" alt=""/><br /><sub><b>Edoardo</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=EdoPro98" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/JMSchoeffmann"><img src="https://avatars1.githubusercontent.com/u/26558330?v=4" width="100px;" alt=""/><br /><sub><b>JMSchoeffmann</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=JMSchoeffmann" title="Code">💻</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/alexander-held"><img src="https://avatars1.githubusercontent.com/u/45009355?v=4" width="100px;" alt=""/><br /><sub><b>alexander-held</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot4/commits?author=alexander-held" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

💻: code, 📖: documentation, 🚇: infrastructure, 🚧: maintainance, ⚠: tests and feedback, 🤔: foundational ideas.
