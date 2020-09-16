.. toctree::
    :caption: Tutorials
    :hidden:

    basic
    uproot3-to-4

.. include:: uproot4.toctree

.. include:: uproot4.reading.toctree

.. include:: uproot4.behaviors.toctree

.. include:: uproot4.model.toctree

.. include:: uproot4.streamers.toctree

.. include:: uproot4.cache.toctree

.. include:: uproot4.compression.toctree

.. include:: uproot4.deserialization.toctree

.. include:: uproot4.source.toctree

.. include:: uproot4.interpretation.toctree

.. include:: uproot4.containers.toctree

.. include:: uproot4.language.toctree

.. include:: uproot4.models.toctree

.. include:: uproot4.const.toctree

.. include:: uproot4.extras.toctree

.. include:: uproot4.version.toctree

.. toctree::
    :hidden:

    uproot4.dynamic

.. |br| raw:: html

    <br/>

.. image:: https://github.com/scikit-hep/uproot4/raw/master/docs-img/logo/logo-300px.png
    :width: 300px
    :alt: Uproot

|br| Uproot is a library for reading (and soon, writing) ROOT files in pure Python and NumPy.

How to install
==============

Usually, you'll want to install Uproot with `Awkward Array <https://awkward-array.org>`__ because this is the default array format.

.. code-block:: bash

    pip install uproot4 awkward1

But if you are working in a limited environment, Uproot can be installed without Awkward Array.

.. code-block:: bash

    pip install uproot4

Just be sure to pass ``library="np"`` to any function that returns arrays to specify that you want NumPy arrays, rather than Awkward arrays. Other array libraries include `Pandas <https://pandas.pydata.org/>`__ and `CuPy <https://cupy.dev/>`__, which, like Awkward, would need to be explicitly installed.

Documentation
=============

**ROOT** is a C++ toolkit for data analysis, part of which is the ROOT file format. Over an exabyte of particle physics data are stored in ROOT files around the world.

**Uproot** is a Python implementation of ROOT I/O, independent of the ROOT toolkit itself (including ROOT's Python interface, PyROOT).

- If you need help understanding ROOT and its ecosystem, see the `ROOT project documentation <https://root.cern/get_started/>`__.
- If you understand what a ROOT file is but have never used Uproot, see :doc:`basic`.
- If you are migrating from an older version to Uproot 4, see the :doc:`uproot3-to-4`.
- If you need detailed descriptions of a class's properties or a function's parameters, see the left-bar on this site (≡ button on mobile) or use ``help`` in Python, ``?`` or shift-tab in iPython/Jupyter.

Getting help
============

Report bugs, request features, and ask for additional documentation on `GitHub Issues <https://github.com/scikit-hep/uproot4/issues>`__.

If you have a problem that's too specific to be new documentation or it isn't exclusively related to Uproot, it might be more appropriate to ask on `StackOverflow with the [uproot] tag <https://stackoverflow.com/questions/tagged/uproot>`__. Be sure to include tags for any other libraries that you use, such as boost-histogram or Pandas.

The `Gitter Scikit-HEP/community <https://gitter.im/Scikit-HEP/community>`__ is a way to get in touch with all Scikit-HEP developers and users.
