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

Getting help
************

Report bugs, request features, and ask for additional documentation on `GitHub Issues <https://github.com/scikit-hep/uproot4/issues>`__. If you have a general "How do I…?" question, we'll answer it as a new example on this site.

If you have a problem that's too specific to be new documentation or it isn't exclusively related to Uproot, it might be more appropriate to ask on `StackOverflow with the [uproot] tag <https://stackoverflow.com/questions/tagged/uproot>`__. Be sure to include tags for any other libraries that you use, such as boost-histogram or Pandas.

If you're migrating from Uproot 3 to Uproot 4, see the `Uproot 3 → 4 cheat-sheet <uproot3-to-4.html>`__.

The `Gitter Scikit-HEP/community <https://gitter.im/Scikit-HEP/community>`__ is a way to get in touch with all Scikit-HEP developers and users.

Basic use
*********

If you've never used Uproot (or Uproot 4) before, start here. Otherwise, see the `Advanced use <#advanced-use>`__ tutorials, or the `Reference <#reference>`__ for each class and function, below. They're also accessible in the left side-bar.

If you're unfamiliar with ROOT, see the `ROOT project's documentation <https://root.cern/get_started/>`__. ROOT is a C++ toolkit for data analysis; the ROOT file format is a part of that ecosystem. ROOT has a Python interface called PyROOT, but be aware that Uproot is an independent Python implementation of the file format only.

Opening a file
""""""""""""""

Open a ROOT file with the :doc:`uproot4.reading.open` function.

.. code-block:: python

    >>> import uproot4
    >>> file = uproot4.open("path/to/dataset.root")

This function can take a string or `pathlib.Path <https://docs.python.org/3/library/pathlib.html>`__, can be used in a ``with`` statement to close the file at the end of the ``with`` block, can point to a local file name, an HTTP URL, or an XRootD ("``root://``") URL. See the documentation for more options.

Finding objects in a file
"""""""""""""""""""""""""

The object returned by :doc:`uproot4.reading.open` represents a TDirectory inside the file (``/``).

.. code-block:: python

    >>> file = uproot4.open("https://scikit-hep.org/uproot/examples/nesteddirs.root")
    >>> file
    <ReadOnlyDirectory '/' at 0x7c070dc03040>

This object is a Python `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__, which means that you can get a list of contents with :doc:`uproot4.reading.ReadOnlyDirectory.keys`.

.. code-block:: python

    >>> file.keys()
    ['one;1', 'one/two;1', 'one/two/tree;1', 'one/tree;1', 'three;1', 'three/tree;1']

and extract an item (read it from the file) with square brackets. The cycle number (after ``;``) doesn't have to be included and you can extract from TDirectories in TDirectories with slashes (``/``).

.. code-block:: python

    >>> file["one"]
    <ReadOnlyDirectory '/one' at 0x78a2045f0fa0>
    >>> file["one"]["two"]
    <ReadOnlyDirectory '/one/two' at 0x78a2045fcca0>
    >>> file["one"]["two"]["tree"]
    <TTree 'tree' (20 branches) at 0x78a2045fcf40>
    >>> file["one/two/tree"]
    <TTree 'tree' (20 branches) at 0x78a2045fcf40>

Data, including nested TDirectories, are not read from disk until they are explicitly requested with square brackets (or another Mapping function, like :doc:`uproot4.reading.ReadOnlyDirectory.values` or :doc:`uproot4.reading.ReadOnlyDirectory.items`).

You can get the names of classes without reading the objects using :doc:`uproot4.reading.ReadOnlyDirectory.classnames`.

.. code-block:: python

    >>> file.classnames()
    {'one': 'TDirectory', 'one/two': 'TDirectory', 'one/two/tree': 'TTree', 'one/tree': 'TTree',
     'three': 'TDirectory', 'three/tree': 'TTree'}

As a shortcut, you can open a file and jump straight to the object by separating the file path and object path with a colon (``:``).

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")
    >>> events
    <TTree 'events' (20 branches) at 0x78e575394b20>

Colon separators are only allowed in strings, so you can open files that have colons in their names by wrapping them in a `pathlib.Path <https://docs.python.org/3/library/pathlib.html>`__.

Inspecting a TBranches of a TTree
"""""""""""""""""""""""""""""""""

ROOT's TTree objects are also `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__ objects with :doc:`uproot4.behaviors.TBranch.HasBranches.keys`, :doc:`uproot4.behaviors.TBranch.HasBranches.values`, and :doc:`uproot4.behaviors.TBranch.HasBranches.items` functions. In this case, the values are TBranch objects (with subbranches accessible through ``/`` paths).

.. code-block:: python

    >>> events.keys()
    ['Type', 'Run', 'Event', 'E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'Q1', 'E2', 'px2',
     'py2', 'pz2', 'pt2', 'eta2', 'phi2', 'Q2', 'M']
    >>> events.values()
    [<TBranch 'Type' at 0x78e575394fa0>, <TBranch 'Run' at 0x78e5753ba730>,
     <TBranch 'Event' at 0x78e5753bae50>, <TBranch 'E1' at 0x78e5753bf5b0>,
     <TBranch 'px1' at 0x78e5753bfcd0>, <TBranch 'py1' at 0x78e574bfc430>,
     <TBranch 'pz1' at 0x78e574bfcb50>, <TBranch 'pt1' at 0x78e574c022b0>,
     <TBranch 'eta1' at 0x78e574c029d0>, <TBranch 'phi1' at 0x78e574c02e80>,
     <TBranch 'Q1' at 0x78e574c08850>, <TBranch 'E2' at 0x78e574c08f70>,
     <TBranch 'px2' at 0x78e574c0c6d0>, <TBranch 'py2' at 0x78e574c0cdf0>,
     <TBranch 'pz2' at 0x78e574c12550>, <TBranch 'pt2' at 0x78e574c12c70>,
     <TBranch 'eta2' at 0x78e574c193d0>, <TBranch 'phi2' at 0x78e574c19af0>,
     <TBranch 'Q2' at 0x78e574c19fa0>, <TBranch 'M' at 0x78e574c1e970>]

Like a TDirectory's classnames, you can access the data types without reading data by calling :doc:`uproot4.behaviors.TBranch.HasBranches.typenames`.

    .. code-block:: python

    >>> events.typenames()
    {'Type': 'char*', 'Run': 'int32_t', 'Event': 'int32_t', 'E1': 'double', 'px1': 'double',
     'py1': 'double', 'pz1': 'double', 'pt1': 'double', 'eta1': 'double', 'phi1': 'double',
     'Q1': 'int32_t', 'E2': 'double', 'px2': 'double', 'py2': 'double', 'pz2': 'double',
     'pt2': 'double', 'eta2': 'double', 'phi2': 'double', 'Q2': 'int32_t', 'M': 'double'}

In an interactive session, 


01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234

Extracting a TBranch as an array
""""""""""""""""""""""""""""""""

Extracting multiple TBranches as a group of arrays
""""""""""""""""""""""""""""""""""""""""""""""""""

Iterating over intervals of entries
"""""""""""""""""""""""""""""""""""

Iterating over many files
"""""""""""""""""""""""""

Reading many files into big arrays
""""""""""""""""""""""""""""""""""

Reading data on demand into lazy arrays
"""""""""""""""""""""""""""""""""""""""

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
