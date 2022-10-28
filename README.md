<img src="https://raw.githubusercontent.com/scikit-hep/uproot4/main/docs-img/logo/logo.svg" width="300px">

[![PyPI version](https://badge.fury.io/py/uproot.svg)](https://pypi.org/project/uproot)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/uproot)](https://github.com/conda-forge/uproot-feedstock)
[![Python 3.7‒3.11](https://img.shields.io/badge/python-3.7%E2%80%923.11-blue)](https://www.python.org)
[![BSD-3 Clause License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Continuous integration tests](https://img.shields.io/github/workflow/status/scikit-hep/uproot4/Test%20build/main?label=tests)](https://github.com/scikit-hep/uproot4/actions)

[![Scikit-HEP](https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg)](https://scikit-hep.org/)
[![NSF-1836650](https://img.shields.io/badge/NSF-1836650-blue.svg)](https://nsf.gov/awardsearch/showAward?AWD_ID=1836650)
[![DOI 10.5281/zenodo.4340632](https://zenodo.org/badge/DOI/10.5281/zenodo.4340632.svg)](https://doi.org/10.5281/zenodo.4340632)
[![Documentation](https://img.shields.io/badge/docs-online-success)](https://uproot.readthedocs.io/)
[![Gitter](https://img.shields.io/badge/chat-online-success)](https://gitter.im/Scikit-HEP/uproot)

Uproot is a library for reading and writing [ROOT files](https://root.cern/) in pure Python and NumPy.

Unlike the standard C++ ROOT implementation, Uproot is only an I/O library, primarily intended to stream data into machine learning libraries in Python. Unlike PyROOT and root_numpy, Uproot does not depend on C++ ROOT. Instead, it uses Numpy to cast blocks of data from the ROOT file as Numpy arrays.

<p align="center"><img src="https://raw.githubusercontent.com/scikit-hep/uproot4/main/docs-img/diagrams/abstraction-layers.svg" width="700px"></p>

# Installation

Uproot can be installed [from PyPI](https://pypi.org/project/uproot) using pip. [Awkward Array](https://pypi.org/project/awkward) is optional but highly recommended:

```bash
pip install uproot awkward
```

Uproot is also available using [conda](https://anaconda.org/conda-forge/uproot) (in this case, [Awkward Array](https://anaconda.org/conda-forge/awkward) is automatically installed):

```bash
conda install -c conda-forge uproot
```

If you have already added `conda-forge` as a channel, the `-c conda-forge` is unnecessary. Adding the channel is recommended because it ensures that all of your packages use compatible versions (see [conda-forge docs](https://conda-forge.org/docs/user/introduction.html#how-can-i-install-packages-from-conda-forge)):

```bash
conda config --add channels conda-forge
conda update --all
```

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

**Uproot's only strict dependencies are NumPy and packaging.** You'll also need `importlib_metadata` if you are on Python 3.7. These will be automatically installed when you install uproot.

**Awkward Array is highly recommended.** It is not a strict dependency, but is included in the dependencies to help most users; install without dependencies (`--no-deps` with pip) if you want to avoid `awkward`. If you're using Uproot without Awkward Array, you'll have to use the `library="np"` option or globally set `uproot.default_library` to return arrays as NumPy arrays (see documentation).

   * `awkward`: be sure to use Awkward Array 2.x.

The following libraries are also useful in conjunction with Uproot, but are not necessary. If you call a function that needs one, you'll be prompted to install it. (Conda installs most of these automatically.)

**For ROOT files, compressed different ways:**

   * `lz4` and `xxhash`: only if reading ROOT files that have been LZ4-compressed.
   * `zstandard`: only if reading ROOT files that have been ZSTD-compressed.

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
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/jpivarski"><img src="https://avatars0.githubusercontent.com/u/1852447?v=4?s=100" width="100px;" alt="Jim Pivarski"/><br /><sub><b>Jim Pivarski</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=jpivarski" title="Code">💻</a> <a href="https://github.com/scikit-hep/uproot5/commits?author=jpivarski" title="Documentation">📖</a> <a href="#infra-jpivarski" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-jpivarski" title="Maintenance">🚧</a></td>
      <td align="center"><a href="https://github.com/reikdas"><img src="https://avatars0.githubusercontent.com/u/11775615?v=4?s=100" width="100px;" alt="Pratyush Das"/><br /><sub><b>Pratyush Das</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=reikdas" title="Code">💻</a> <a href="#infra-reikdas" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center"><a href="https://github.com/chrisburr"><img src="https://avatars3.githubusercontent.com/u/5220533?v=4?s=100" width="100px;" alt="Chris Burr"/><br /><sub><b>Chris Burr</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=chrisburr" title="Code">💻</a> <a href="#infra-chrisburr" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center"><a href="https://github.com/plexoos"><img src="https://avatars0.githubusercontent.com/u/5005079?v=4?s=100" width="100px;" alt="Dmitri Smirnov"/><br /><sub><b>Dmitri Smirnov</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=plexoos" title="Code">💻</a></td>
      <td align="center"><a href="http://www.matthewfeickert.com/"><img src="https://avatars3.githubusercontent.com/u/5142394?v=4?s=100" width="100px;" alt="Matthew Feickert"/><br /><sub><b>Matthew Feickert</b></sub></a><br /><a href="#infra-matthewfeickert" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center"><a href="http://www.tamasgal.com"><img src="https://avatars1.githubusercontent.com/u/1730350?v=4?s=100" width="100px;" alt="Tamas Gal"/><br /><sub><b>Tamas Gal</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=tamasgal" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/kreczko"><img src="https://avatars3.githubusercontent.com/u/1213276?v=4?s=100" width="100px;" alt="Luke Kreczko"/><br /><sub><b>Luke Kreczko</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=kreczko" title="Code">💻</a> <a href="https://github.com/scikit-hep/uproot5/commits?author=kreczko" title="Tests">⚠️</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/nsmith-"><img src="https://avatars2.githubusercontent.com/u/6587412?v=4?s=100" width="100px;" alt="Nicholas Smith"/><br /><sub><b>Nicholas Smith</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=nsmith-" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/nbiederbeck"><img src="https://avatars1.githubusercontent.com/u/15156697?v=4?s=100" width="100px;" alt="Noah Biederbeck"/><br /><sub><b>Noah Biederbeck</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=nbiederbeck" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/oshadura"><img src="https://avatars2.githubusercontent.com/u/7012420?v=4?s=100" width="100px;" alt="Oksana Shadura"/><br /><sub><b>Oksana Shadura</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=oshadura" title="Code">💻</a> <a href="#infra-oshadura" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center"><a href="http://iscinumpy.gitlab.io"><img src="https://avatars1.githubusercontent.com/u/4616906?v=4?s=100" width="100px;" alt="Henry Schreiner"/><br /><sub><b>Henry Schreiner</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=henryiii" title="Code">💻</a> <a href="#infra-henryiii" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/scikit-hep/uproot5/commits?author=henryiii" title="Tests">⚠️</a></td>
      <td align="center"><a href="https://github.com/masonproffitt"><img src="https://avatars3.githubusercontent.com/u/32773304?v=4?s=100" width="100px;" alt="Mason Proffitt"/><br /><sub><b>Mason Proffitt</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=masonproffitt" title="Code">💻</a> <a href="https://github.com/scikit-hep/uproot5/commits?author=masonproffitt" title="Tests">⚠️</a></td>
      <td align="center"><a href="https://www.linkedin.com/in/jonas-rembser/"><img src="https://avatars2.githubusercontent.com/u/6578603?v=4?s=100" width="100px;" alt="Jonas Rembser"/><br /><sub><b>Jonas Rembser</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=guitargeek" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/benkrikler"><img src="https://avatars0.githubusercontent.com/u/4083697?v=4?s=100" width="100px;" alt="benkrikler"/><br /><sub><b>benkrikler</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=benkrikler" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/HDembinski"><img src="https://avatars0.githubusercontent.com/u/2631586?v=4?s=100" width="100px;" alt="Hans Dembinski"/><br /><sub><b>Hans Dembinski</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=HDembinski" title="Documentation">📖</a></td>
      <td align="center"><a href="http://marcelrieger.com"><img src="https://avatars0.githubusercontent.com/u/1908734?v=4?s=100" width="100px;" alt="Marcel R."/><br /><sub><b>Marcel R.</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=riga" title="Code">💻</a></td>
      <td align="center"><a href="http://turra.web.cern.ch/turra/"><img src="https://avatars3.githubusercontent.com/u/143389?v=4?s=100" width="100px;" alt="Ruggero Turra"/><br /><sub><b>Ruggero Turra</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=wiso" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/jrueb"><img src="https://avatars2.githubusercontent.com/u/30041073?v=4?s=100" width="100px;" alt="Jonas Rübenach"/><br /><sub><b>Jonas Rübenach</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=jrueb" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/bfis"><img src="https://avatars0.githubusercontent.com/u/15651150?v=4?s=100" width="100px;" alt="bfis"/><br /><sub><b>bfis</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=bfis" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/raymondEhlers"><img src="https://avatars0.githubusercontent.com/u/1571927?v=4?s=100" width="100px;" alt="Raymond Ehlers"/><br /><sub><b>Raymond Ehlers</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=raymondEhlers" title="Code">💻</a></td>
      <td align="center"><a href="http://andrzejnovak.github.io/"><img src="https://avatars1.githubusercontent.com/u/13226500?v=4?s=100" width="100px;" alt="Andrzej Novak"/><br /><sub><b>Andrzej Novak</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=andrzejnovak" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/bendavid"><img src="https://avatars2.githubusercontent.com/u/4920798?v=4?s=100" width="100px;" alt="Josh Bendavid"/><br /><sub><b>Josh Bendavid</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=bendavid" title="Code">💻</a></td>
      <td align="center"><a href="https://ddavis.io/"><img src="https://avatars2.githubusercontent.com/u/3202090?v=4?s=100" width="100px;" alt="Doug Davis"/><br /><sub><b>Doug Davis</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=douglasdavis" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/asymmetry"><img src="https://avatars3.githubusercontent.com/u/679529?v=4?s=100" width="100px;" alt="Chao Gu"/><br /><sub><b>Chao Gu</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=asymmetry" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/ast0815"><img src="https://avatars2.githubusercontent.com/u/5884065?v=4?s=100" width="100px;" alt="Lukas Koch"/><br /><sub><b>Lukas Koch</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=ast0815" title="Code">💻</a></td>
      <td align="center"><a href="http://irfu.cea.fr/dap/"><img src="https://avatars1.githubusercontent.com/u/17836610?v=4?s=100" width="100px;" alt="Michele Peresano"/><br /><sub><b>Michele Peresano</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=HealthyPear" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/EdoPro98"><img src="https://avatars1.githubusercontent.com/u/57357892?v=4?s=100" width="100px;" alt="Edoardo"/><br /><sub><b>Edoardo</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=EdoPro98" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/JMSchoeffmann"><img src="https://avatars1.githubusercontent.com/u/26558330?v=4?s=100" width="100px;" alt="JMSchoeffmann"/><br /><sub><b>JMSchoeffmann</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=JMSchoeffmann" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/alexander-held"><img src="https://avatars1.githubusercontent.com/u/45009355?v=4?s=100" width="100px;" alt="alexander-held"/><br /><sub><b>alexander-held</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=alexander-held" title="Code">💻</a></td>
      <td align="center"><a href="https://giordonstark.com/"><img src="https://avatars0.githubusercontent.com/u/761483?v=4?s=100" width="100px;" alt="Giordon Stark"/><br /><sub><b>Giordon Stark</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=kratsg" title="Code">💻</a></td>
      <td align="center"><a href="https://roneil.xyz"><img src="https://avatars.githubusercontent.com/u/56410978?v=4?s=100" width="100px;" alt="Ryunosuke O'Neil"/><br /><sub><b>Ryunosuke O'Neil</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=ryuwd" title="Code">💻</a></td>
      <td align="center"><a href="https://christopherappold.netlify.app/index.html"><img src="https://avatars.githubusercontent.com/u/28101201?v=4?s=100" width="100px;" alt="ChristopheRappold"/><br /><sub><b>ChristopheRappold</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=ChristopheRappold" title="Documentation">📖</a></td>
      <td align="center"><a href="http://cozzyd.net"><img src="https://avatars.githubusercontent.com/u/9206569?v=4?s=100" width="100px;" alt="Cosmin Deaconu"/><br /><sub><b>Cosmin Deaconu</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=cozzyd" title="Tests">⚠️</a> <a href="https://github.com/scikit-hep/uproot5/commits?author=cozzyd" title="Code">💻</a></td>
      <td align="center"><a href="https://peguerosdc.github.io/"><img src="https://avatars.githubusercontent.com/u/7889726?v=4?s=100" width="100px;" alt="Carlos Pegueros"/><br /><sub><b>Carlos Pegueros</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=peguerosdc" title="Documentation">📖</a> <a href="#example-peguerosdc" title="Examples">💡</a> <a href="https://github.com/scikit-hep/uproot5/commits?author=peguerosdc" title="Tests">⚠️</a> <a href="#tutorial-peguerosdc" title="Tutorials">✅</a></td>
      <td align="center"><a href="https://github.com/btovar"><img src="https://avatars.githubusercontent.com/u/3081826?v=4?s=100" width="100px;" alt="Benjamin Tovar"/><br /><sub><b>Benjamin Tovar</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=btovar" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://duncanmmacleod.github.io"><img src="https://avatars.githubusercontent.com/u/1618530?v=4?s=100" width="100px;" alt="Duncan Macleod"/><br /><sub><b>Duncan Macleod</b></sub></a><br /><a href="#infra-duncanmmacleod" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center"><a href="https://github.com/mpad"><img src="https://avatars.githubusercontent.com/u/1219868?v=4?s=100" width="100px;" alt="mpad"/><br /><sub><b>mpad</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=mpad" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/pfackeldey"><img src="https://avatars.githubusercontent.com/u/18463582?v=4?s=100" width="100px;" alt="Peter Fackeldey"/><br /><sub><b>Peter Fackeldey</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=pfackeldey" title="Code">💻</a></td>
      <td align="center"><a href="http://blog.kushkothari.in"><img src="https://avatars.githubusercontent.com/u/53650538?v=4?s=100" width="100px;" alt="Kush Kothari"/><br /><sub><b>Kush Kothari</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=kkothari2001" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/aryan26roy"><img src="https://avatars.githubusercontent.com/u/50577809?v=4?s=100" width="100px;" alt="Aryan Roy"/><br /><sub><b>Aryan Roy</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=aryan26roy" title="Code">💻</a></td>
      <td align="center"><a href="http://blog.jling.dev"><img src="https://avatars.githubusercontent.com/u/5306213?v=4?s=100" width="100px;" alt="Jerry Ling"/><br /><sub><b>Jerry Ling</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=Moelf" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/kakwok"><img src="https://avatars.githubusercontent.com/u/12798013?v=4?s=100" width="100px;" alt="kakwok"/><br /><sub><b>kakwok</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=kakwok" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center"><a href="https://github.com/veprbl"><img src="https://avatars.githubusercontent.com/u/245573?v=4?s=100" width="100px;" alt="Dmitry Kalinkin"/><br /><sub><b>Dmitry Kalinkin</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=veprbl" title="Code">💻</a> <a href="#infra-veprbl" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center"><a href="https://github.com/nikoladze"><img src="https://avatars.githubusercontent.com/u/3707225?v=4?s=100" width="100px;" alt="Nikolai Hartmann"/><br /><sub><b>Nikolai Hartmann</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=nikoladze" title="Code">💻</a></td>
      <td align="center"><a href="https://www.lieret.net"><img src="https://avatars.githubusercontent.com/u/13602468?v=4?s=100" width="100px;" alt="Kilian Lieret"/><br /><sub><b>Kilian Lieret</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=klieret" title="Documentation">📖</a></td>
      <td align="center"><a href="https://github.com/dcervenkov"><img src="https://avatars.githubusercontent.com/u/23052054?v=4?s=100" width="100px;" alt="Daniel Cervenkov"/><br /><sub><b>Daniel Cervenkov</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=dcervenkov" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/beojan"><img src="https://avatars.githubusercontent.com/u/3727925?v=4?s=100" width="100px;" alt="Beojan Stanislaus"/><br /><sub><b>Beojan Stanislaus</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=beojan" title="Code">💻</a></td>
      <td align="center"><a href="https://github.com/agoose77"><img src="https://avatars.githubusercontent.com/u/1248413?v=4?s=100" width="100px;" alt="Angus Hollands"/><br /><sub><b>Angus Hollands</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=agoose77" title="Code">💻</a> <a href="#maintenance-agoose77" title="Maintenance">🚧</a></td>
      <td align="center"><a href="https://github.com/lobis"><img src="https://avatars.githubusercontent.com/u/35803280?v=4?s=100" width="100px;" alt="Luis Antonio Obis Aparicio"/><br /><sub><b>Luis Antonio Obis Aparicio</b></sub></a><br /><a href="https://github.com/scikit-hep/uproot5/commits?author=lobis" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

💻: code, 📖: documentation, 🚇: infrastructure, 🚧: maintainance, ⚠: tests/feedback, 🤔: foundational ideas.
