Basic use: start here!
======================

Opening a file
--------------

Open a ROOT file with the :py:func:`~uproot4.reading.open` function.

.. code-block:: python

    >>> import uproot4
    >>> file = uproot4.open("path/to/dataset.root")

This function can take a string or `pathlib.Path <https://docs.python.org/3/library/pathlib.html>`__, can be used in a ``with`` statement to close the file at the end of the ``with`` block, can point to a local file name, an HTTP URL, or an XRootD ("``root://``") URL. See the documentation for more options.

Finding objects in a file
-------------------------

The object returned by :py:func:`~uproot4.reading.open` represents a TDirectory inside the file (``/``).

.. code-block:: python

    >>> file = uproot4.open("https://scikit-hep.org/uproot/examples/nesteddirs.root")
    >>> file
    <ReadOnlyDirectory '/' at 0x7c070dc03040>

This object is a Python `Mapping <https://docs.python.org/3/library/stdtypes.html-mapping-types-dict>`__, which means that you can get a list of contents with :py:meth:`uproot4.reading.ReadOnlyDirectory.keys`.

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

Data, including nested TDirectories, are not read from disk until they are explicitly requested with square brackets (or another Mapping function, like :py:meth:`~uproot4.reading.ReadOnlyDirectory.values` or :py:meth:`~uproot4.reading.ReadOnlyDirectory.items`).

You can get the names of classes without reading the objects using :py:meth:`~uproot4.reading.ReadOnlyDirectory.classnames`.

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
---------------------------------

ROOT's TTree objects are also `Mapping <https://docs.python.org/3/library/stdtypes.html-mapping-types-dict>`__ objects with :py:meth:`~uproot4.behaviors.TBranch.HasBranches.keys`, :py:meth:`~uproot4.behaviors.TBranch.HasBranches.values`, and :py:meth:`~uproot4.behaviors.TBranch.HasBranches.items` functions. In this case, the values are TBranch objects (with subbranches accessible through ``/`` paths).

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")

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

    >>> events["M"]
    <TBranch 'M' at 0x78e574c1e970>

Like a TDirectory's classnames, you can access the data types without reading data by calling :py:meth:`~uproot4.behaviors.TBranch.HasBranches.typenames`.

.. code-block:: python

    >>> events.typenames()
    {'Type': 'char*', 'Run': 'int32_t', 'Event': 'int32_t', 'E1': 'double', 'px1': 'double',
     'py1': 'double', 'pz1': 'double', 'pt1': 'double', 'eta1': 'double', 'phi1': 'double',
     'Q1': 'int32_t', 'E2': 'double', 'px2': 'double', 'py2': 'double', 'pz2': 'double',
     'pt2': 'double', 'eta2': 'double', 'phi2': 'double', 'Q2': 'int32_t', 'M': 'double'}

In an interactive session, it's often more convenient to call :py:meth:`~uproot4.behaviors.TBranch.HasBranches.show`.

.. code-block:: python

    >>> events.show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    Type                 | char*                    | AsStrings()
    Run                  | int32_t                  | AsDtype('>i4')
    Event                | int32_t                  | AsDtype('>i4')
    E1                   | double                   | AsDtype('>f8')
    px1                  | double                   | AsDtype('>f8')
    py1                  | double                   | AsDtype('>f8')
    pz1                  | double                   | AsDtype('>f8')
    pt1                  | double                   | AsDtype('>f8')
    eta1                 | double                   | AsDtype('>f8')
    phi1                 | double                   | AsDtype('>f8')
    Q1                   | int32_t                  | AsDtype('>i4')
    E2                   | double                   | AsDtype('>f8')
    px2                  | double                   | AsDtype('>f8')
    py2                  | double                   | AsDtype('>f8')
    pz2                  | double                   | AsDtype('>f8')
    pt2                  | double                   | AsDtype('>f8')
    eta2                 | double                   | AsDtype('>f8')
    phi2                 | double                   | AsDtype('>f8')
    Q2                   | int32_t                  | AsDtype('>i4')
    M                    | double                   | AsDtype('>f8')

The third column, ``interpretation``, indicates how data in the TBranch will be interpreted as an array.

Reading a TBranch as an array
-----------------------------

A TBranch may be turned into an array with the :py:meth:`~uproot4.behavior.TBranch.TBranch.array` method. The array is not read from disk until this method (or equivalent) is called.

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")
    >>> events["M"].array()
    <Array [82.5, 83.6, 83.3, ... 96, 96.5, 96.7] type='2304 * float64'>

By default, the array is an Awkward array, as shown above. This assumes that Awkward Array is installed (see `How to install <-how-to-install>`__). If you can't install it or want to use NumPy for other reasons, pass ``library="np"`` instead of the default ``library="ak"``.

.. code-block:: python

    >>> events["M"].array(library="np")
    array([82.46269156, 83.62620401, 83.30846467, ..., 95.96547966,
           96.49594381, 96.65672765])

Another library option is ``library="pd"`` for Pandas, and a single TBranch is (usually) presented as a `pandas.Series <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html>`__.

.. code-block:: python

    >>> events["M"].array(library="pd")
    0       82.462692
    1       83.626204
    2       83.308465
    3       82.149373
    4       90.469123
              ...    
    2299    60.047138
    2300    96.125376
    2301    95.965480
    2302    96.495944
    2303    96.656728
    Length: 2304, dtype: float64

If you don't have the specified library (including the default, Awkward), you'll be prompted with instructions to install it.

.. code-block:: python

    >>> events["M"].array(library="cp")
    Traceback (most recent call last):
      File "/home/jpivarski/irishep/uproot4/uproot4/extras.py", line 60, in cupy
        import cupy
    ModuleNotFoundError: No module named 'cupy'

    ...

    ImportError: install the 'cupy' package with:

        pip install cupy

    or

        conda install cupy

(CuPy can only be used on computers with GPUs.)

The :py:meth:`~uproot4.behavior.TBranch.TBranch.array` method has many options, including limitations on reading (``entry_start`` and ``entry_stop``), parallelization (``decompression_executor`` and ``interpretation_executor``), and caching (``array_cache``). See the documentation for details.

Reading multiple TBranches as a group of arrays
-----------------------------------------------

To read more than one TBranch, you could use the :py:meth:`~uproot4.behavior.TBranch.TBranch.array` method from the previous section multiple times, but you could also use :py:meth:`~uproot4.behavior.TBranch.HasBranches.arrays` (plural) on the TTree itself.

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")

    >>> momentum = events.arrays(["px1", "py1", "pz1"])
    >>> momentum
    <Array [{px1: -41.2, ... pz1: -74.8}] type='2304 * {"px1": float64, "py1": float...'>

The return value is a group of arrays, where a "group" has different meanings in different libraries. For Awkward Array (above), a group is an array of records, which can be projected like this:

.. code-block:: python

    >>> momentum["px1"]
    <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>

For NumPy, a group is a dict of arrays.

.. code-block:: python

    >>> momentum = events.arrays(["px1", "py1", "pz1"], library="np")
    >>> momentum
    {'px1': array([-41.19528764,  35.11804977,  35.11804977, ...,  32.37749196,
            32.37749196,  32.48539387]),
     'py1': array([ 17.4332439 , -16.57036233, -16.57036233, ...,   1.19940578,
             1.19940578,   1.2013503 ]),
     'pz1': array([-68.96496181, -48.77524654, -48.77524654, ..., -74.53243061,
           -74.53243061, -74.80837247])}
    >>> momentum["px1"]
    array([-41.19528764,  35.11804977,  35.11804977, ...,  32.37749196,
            32.37749196,  32.48539387])

For Pandas, a group is a `pandas.DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`__.

.. code-block:: python

    >>> momentum = events.arrays(["px1", "py1", "pz1"], library="pd")
    >>> momentum
                px1        py1         pz1
    0    -41.195288  17.433244  -68.964962
    1     35.118050 -16.570362  -48.775247
    2     35.118050 -16.570362  -48.775247
    3     34.144437 -16.119525  -47.426984
    4     22.783582  15.036444  -31.689894
    ...         ...        ...         ...
    2299  19.054651  14.833954   22.051323
    2300 -68.041915 -26.105847 -152.235018
    2301  32.377492   1.199406  -74.532431
    2302  32.377492   1.199406  -74.532431
    2303  32.485394   1.201350  -74.808372

    [2304 rows x 3 columns]

    >>> momentum["px1"]
    0      -41.195288
    1       35.118050
    2       35.118050
    3       34.144437
    4       22.783582
              ...    
    2299    19.054651
    2300   -68.041915
    2301    32.377492
    2302    32.377492
    2303    32.485394
    Name: px1, Length: 2304, dtype: float64

Choosing TBranches to read
--------------------------

If no arguments are passed to :py:meth:`~uproot4.behavior.TBranch.HasBranches.arrays`, *all* TBranches will be read. If your file has many TBranches, this might not be desirable or possible. You can select specific TBranches by name, as in the previous section, but you can also pass a filter (``filter_name``, ``filter_typename``, or ``filter_branch``):

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")

    >>> events.keys(filter_name="px*")
    ['px1', 'px2']
    >>> events.arrays(filter_name="px*")
    <Array [{px1: -41.2, ... px2: -68.8}] type='2304 * {"px1": float64, "px2": float64}'>

    >>> events.keys(filter_name="/p[xyz][0-9]/i")
    ['px1', 'py1', 'pz1', 'px2', 'py2', 'pz2']
    >>> events.arrays(filter_name="/p[xyz][0-9]/i")
    <Array [{px1: -41.2, py1: 17.4, ... pz2: -154}] type='2304 * {"px1": float64, "p...'>

    >>> events.keys(filter_branch=lambda b: b.compression_ratio > 10)
    ['Run', 'Q1', 'Q2']
    >>> events.arrays(filter_branch=lambda b: b.compression_ratio > 10)
    <Array [{Run: 148031, Q1: 1, ... Q2: -1}] type='2304 * {"Run": int32, "Q1": int3...'>

The first argument, which we used in the previous section to pass explicit TBranch names,

.. code-block:: python

    >>> events.arrays(["px1", "py1", "pz1"])
    <Array [{px1: -41.2, ... pz1: -74.8}] type='2304 * {"px1": float64, "py1": float...'>

can also compute arbitrary expressions.

.. code-block:: python

    >>> events.arrays("sqrt(px1**2 + py1**2)")
    <Array [{'sqrt(px1**2 + py1**2)': 44.7, ... ] type='2304 * {"sqrt(px1**2 + py1**...'>

If the TTree has any aliases, you can refer to these aliases by name, or you can create new aliases to give better names to the keys of the NumPy dict, Awkward records, or Pandas columns.

.. code-block:: python

    >>> events.arrays("pt1", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{pt1: 44.7}, ... {pt1: 32.4}] type='2304 * {"pt1": float64}'>

The second argument (must be a string, not a list of strings) is a ``cut``, or filter on entries.

.. code-block:: python

    >>> events.arrays(["M"], "pt1 > 50", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{M: 91.8}, {M: 91.9, ... {M: 96.1}] type='290 * {"M": float64}'>

Note that expressions are *not*, in general, computed more quickly if expressed in these strings. The above is equivalent to the following

language is Python


Nested data structures
----------------------

Iterating over intervals of entries
-----------------------------------

Iterating over many files
-------------------------

Reading many files into big arrays
----------------------------------

Reading on demand with lazy arrays
----------------------------------
