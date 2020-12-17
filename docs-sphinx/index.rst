.. toctree::
    :hidden:

    changelog

.. toctree::
    :caption: Tutorials
    :hidden:

    basic
    uproot3-to-4

.. include:: uproot.toctree

.. toctree::
    :caption: Modules
    :hidden:

.. include:: uproot.reading.toctree

.. include:: uproot.behaviors.toctree

.. include:: uproot.model.toctree

.. include:: uproot.streamers.toctree

.. include:: uproot.cache.toctree

.. include:: uproot.compression.toctree

.. include:: uproot.deserialization.toctree

.. include:: uproot.source.toctree

.. include:: uproot.interpretation.toctree

.. include:: uproot.containers.toctree

.. include:: uproot.language.toctree

.. include:: uproot.models.toctree

.. include:: uproot.exceptions.toctree

.. |br| raw:: html

    <br/>

.. image:: https://github.com/scikit-hep/uproot4/raw/main/docs-img/logo/logo-300px.png
    :width: 300px
    :alt: Uproot
    :target: https://github.com/scikit-hep/uproot4

.. role:: raw-html(raw)
    :format: html

|br|

:raw-html:`<p>`

.. image:: https://badge.fury.io/py/uproot.svg
   :alt: PyPI version
   :target: https://pypi.org/project/uproot

.. image:: https://img.shields.io/conda/vn/conda-forge/uproot
   :alt: Conda-Forge
   :target: https://github.com/conda-forge/uproot-feedstock

.. image:: https://img.shields.io/badge/Python-3.5%E2%80%923.9-blue
   :alt: Python 3.5‒3.9
   :target: https://www.python.org

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
   :alt: BSD-3 Clause License
   :target: https://opensource.org/licenses/BSD-3-Clause

.. image:: https://img.shields.io/github/workflow/status/scikit-hep/uproot4/Test%20build/main?label=CI
   :alt: Test build
   :target: https://github.com/scikit-hep/uproot4/actions?query=workflow%3A%22Test+build%22

.. image:: https://img.shields.io/github/workflow/status/scikit-hep/uproot4/Deploy%20to%20PyPI?label=CD
   :alt: Deploy to PyPI
   :target: https://github.com/scikit-hep/uproot4/actions?query=workflow%3A%22Deploy+to+PyPI%22

:raw-html:`</p><p>`

.. image:: https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg
   :alt: Scikit-HEP
   :target: https://scikit-hep.org/

.. image:: https://img.shields.io/badge/NSF-1836650-blue.svg
   :alt: NSF-1836650
   :target: https://nsf.gov/awardsearch/showAward?AWD_ID=1836650

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4340632.svg
   :alt: DOI 10.5281/zenodo.4340632
   :target: https://doi.org/10.5281/zenodo.4340632

.. image:: https://img.shields.io/badge/docs-online-success
   :alt: Documentation
   :target: https://uproot.readthedocs.io/

.. image:: https://img.shields.io/badge/chat-online-success
   :alt: Gitter
   :target: https://gitter.im/Scikit-HEP/uproot

:raw-html:`</p>`

|br| Uproot is a library for reading (and soon, writing) ROOT files in pure Python and NumPy.

How to install
==============

Usually, you'll want to install Uproot with `Awkward Array <https://awkward-array.org>`__ because this is the default array format.

.. code-block:: bash

    pip install uproot awkward

But if you are working in a limited environment, Uproot can be installed without Awkward Array.

.. code-block:: bash

    pip install uproot

Just be sure to pass ``library="np"`` to any function that returns arrays or globally set ``uproot.default_library`` to specify that you want NumPy arrays, rather than Awkward Arrays. Other array libraries include `Pandas <https://pandas.pydata.org/>`__ and `CuPy <https://cupy.dev/>`__, which, like Awkward Array, would need to be explicitly installed.

Documentation
=============

**ROOT** is a C++ toolkit for data analysis, part of which is the ROOT file format. Over an exabyte of particle physics data are stored in ROOT files around the world.

**Uproot** is a Python implementation of ROOT I/O, independent of the ROOT toolkit itself (including ROOT's Python interface, PyROOT).

- If you need help understanding ROOT and its ecosystem, see the `ROOT project documentation <https://root.cern/get_started/>`__.
- If you know what a ROOT file is but are unfamiliar with Uproot, see the :doc:`basic`.
- If you are migrating from an older version to Uproot 4, see the :doc:`uproot3-to-4`.
- If you need detailed descriptions of a class's properties or a function's parameters, see the left-bar on this site (≡ button on mobile) or use ``help`` in Python, ``?`` or shift-tab in iPython/Jupyter.

Getting help
============

- Report bugs, request features, and ask for additional documentation on `GitHub Issues <https://github.com/scikit-hep/uproot4/issues>`__.
- If you have a "How do I...?" question, start a `GitHub Discussion <https://github.com/scikit-hep/uproot4/discussions>`__ with category "Q&A".
- Alternatively, ask about it on `StackOverflow with the [uproot] tag <https://stackoverflow.com/questions/tagged/uproot>`__. Be sure to include tags for any other libraries that you use, such as Pandas or PyTorch.
- To ask questions in real time, try the Gitter `Scikit-HEP/uproot <https://gitter.im/Scikit-HEP/uproot>`__ chat room.
