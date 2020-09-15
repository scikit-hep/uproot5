.. |br| raw:: html

    <br/>

.. image:: https://github.com/scikit-hep/uproot4/raw/master/docs-img/logo/logo-300px.png
    :width: 300px
    :alt: Uproot

|br| Uproot is a library for reading (and soon, writing) ROOT files in pure Python and NumPy.

How to install
**************

Usually, you'll want to install Uproot with `Awkward Array <https://awkward-array.org>`__ because this is the default array format.

.. code-block:: bash

    pip install uproot4 awkward1

But if you are working in a limited environment, Uproot can be installed without Awkward Array.

.. code-block:: bash

    pip install uproot4

Just be sure to pass ``library="np"`` to any function that returns arrays to specify that you want NumPy arrays, rather than Awkward arrays. Other array libraries include `Pandas <https://pandas.pydata.org/>`__ and `CuPy <https://cupy.dev/>`__, which, like Awkward, would need to be explicitly installed.

Basic use
*********

FIXME: some examples inline

Advanced use
************

FIXME: a list of tutorials

Reference
*********

FIXME: a list of modules

FIXME: also, the toctree

.. toctree::
    :hidden:

    uproot3-to-4.rst
