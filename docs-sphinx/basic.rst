Basic use: start here!
======================

Opening a file
--------------

Open a ROOT file for reading with the `uproot4.open <uproot4.reading.open.html>`__ function.

.. code-block:: python

    >>> import uproot4
    >>> file = uproot4.open("path/to/dataset.root")

The `uproot4.open <uproot4.reading.open.html>`__ function can also be used like this:

.. code-block:: python

    >>> with uproot4.open("path/to/dataset.root") as file:
    ...     do_something...

to automatically close the file after leaving the ``with`` block. The path-name argument can be a local file (as above), a URL ("``http://``" or "``https://``"), or XRootD ("``root://``") if you have the `Python interface to XRootD <https://anaconda.org/conda-forge/xrootd>`__ installed. It can also be a Python file-like object with ``read`` and ``seek`` methods, but such objects can't be read in parallel.

The `uproot4.open <uproot4.reading.open.html>`__ function has many options, including alternate handlers for each input type, ``num_workers`` to control parallel reading, and caches (``object_cache`` and ``array_cache``). The defaults attempt to optimize parallel processing, caching, and batching of remote requests, but better performance can often be obtained by tuning these parameters.

Finding objects in a file
-------------------------

The object returned by `uproot4.open <uproot4.reading.open.html>`__ represents a TDirectory inside the file (``/``).

.. code-block:: python

    >>> file = uproot4.open("https://scikit-hep.org/uproot/examples/nesteddirs.root")
    >>> file
    <ReadOnlyDirectory '/' at 0x7c070dc03040>

This object is a Python `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__, which means that you can get a list of contents with :py:meth:`~uproot4.reading.ReadOnlyDirectory.keys`.

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

You can get the names of classes without reading the objects by using :py:meth:`~uproot4.reading.ReadOnlyDirectory.classnames`.

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

ROOT's TTree objects are also `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__ objects with :py:meth:`~uproot4.behaviors.TBranch.HasBranches.keys`, :py:meth:`~uproot4.behaviors.TBranch.HasBranches.values`, and :py:meth:`~uproot4.behaviors.TBranch.HasBranches.items` functions. In this case, the values are TBranch objects (with subbranches accessible through ``/`` paths).

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

A TBranch may be turned into an array with the :py:meth:`~uproot4.behaviors.TBranch.TBranch.array` method. The array is not read from disk until this method is called (or other array-fetching methods described below).

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")
    >>> events["M"].array()
    <Array [82.5, 83.6, 83.3, ... 96, 96.5, 96.7] type='2304 * float64'>

By default, the array is an Awkward array, as shown above. This assumes that Awkward Array is installed (see `How to install <index.html#how-to-install>`__). If you can't install it or want to use NumPy for other reasons, pass ``library="np"`` instead of the default ``library="ak"``.

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

The :py:meth:`~uproot4.behaviors.TBranch.TBranch.array` method has many options, including limitations on reading (``entry_start`` and ``entry_stop``), parallelization (``decompression_executor`` and ``interpretation_executor``), and caching (``array_cache``). For details, see the reference documentation for :py:meth:`~uproot4.behaviors.TBranch.TBranch.array`.

Reading multiple TBranches as a group of arrays
-----------------------------------------------

To read more than one TBranch, you could use the :py:meth:`~uproot4.behaviors.TBranch.TBranch.array` method from the previous section multiple times, but you could also use :py:meth:`~uproot4.behaviors.TBranch.HasBranches.arrays` (plural) on the TTree itself.

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

Even though you can extract individual arrays from these objects, they're read, decompressed, and interpreted as soon as you ask for them. Unless you're working with small files, be sure not to read everything when you only want a few of the arrays!

Filtering TBranches
-------------------

If no arguments are passed to :py:meth:`~uproot4.behaviors.TBranch.HasBranches.arrays`, *all* TBranches will be read. If your file has many TBranches, this might not be desirable or possible. You can select specific TBranches by name, as in the previous section, but you can also use a filter (``filter_name``, ``filter_typename``, or ``filter_branch``) to select TBranches by name, type, or other attributes.

The :py:meth:`~uproot4.behaviors.TBranch.HasBranches.keys`, :py:meth:`~uproot4.behaviors.TBranch.HasBranches.values`, :py:meth:`~uproot4.behaviors.TBranch.HasBranches.items`, and :py:meth:`~uproot4.behaviors.TBranch.HasBranches.typenames` methods take the same arguments, so you can test your filters before reading any data.

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

Computing expressions and cuts
------------------------------

The first argument of :py:meth:`~uproot4.behaviors.TBranch.HasBranches.arrays`, which we used above to pass explicit TBranch names,

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")

    >>> events.arrays(["px1", "py1", "pz1"])
    <Array [{px1: -41.2, ... pz1: -74.8}] type='2304 * {"px1": float64, "py1": float...'>

can also compute expressions:

.. code-block:: python

    >>> events.arrays("sqrt(px1**2 + py1**2)")
    <Array [{'sqrt(px1**2 + py1**2)': 44.7, ... ] type='2304 * {"sqrt(px1**2 + py1**...'>

If the TTree has any aliases, you can refer to those aliases by name, or you can create new aliases to give better names to the keys of the output dict, Awkward records, or Pandas columns.

.. code-block:: python

    >>> events.arrays("pt1", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{pt1: 44.7}, ... {pt1: 32.4}] type='2304 * {"pt1": float64}'>

The second argument is a ``cut``, or filter on entries. Whereas the uncut array (above) has 2304 entries, the cut array (below) has 290 entries.

.. code-block:: python

    >>> events.arrays(["M"], "pt1 > 50", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{M: 91.8}, {M: 91.9, ... {M: 96.1}] type='290 * {"M": float64}'>

Note that expressions are *not*, in general, computed more quickly if expressed in these strings. The above is equivalent to the following:

.. code-block:: python

    >>> import numpy as np
    >>> arrays = events.arrays(["px1", "py1", "M"])
    >>> pt1 = np.sqrt(arrays.px1**2 + arrays.py1**2)
    >>> arrays.M[pt1 > 50]
    <Array [91.8, 91.9, 91.7, ... 90.1, 90.1, 96.1] type='289 * float64'>

but perhaps more convenient. If what you want to compute requires more than one expression, you'll have to move it out of strings into Python.

The default ``language`` is :py:class:`~uproot4.language.python.PythonLanguage`, but other languages, like ROOT's `TTree::Draw syntax <https://root.cern.ch/doc/master/classTTree.html#a73450649dc6e54b5b94516c468523e45>`_ are foreseen *in the future*. Thus, implicit loops (e.g. ``Sum$(...)``) have to be translated to their Awkward equivalents and ``ROOT::Math`` functions have to be translated to their NumPy equivalents.

Nested data structures
----------------------

Not all datasets have one value per entry. In particle physics, we often have different numbers of particles (and particle attributes) per collision event.

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/HZZ.root:events")
    >>> events.show()
    name                 | typename                 | interpretation                
    ---------------------+--------------------------+-------------------------------
    NJet                 | int32_t                  | AsDtype('>i4')
    Jet_Px               | float[]                  | AsJagged(AsDtype('>f4'))
    Jet_Py               | float[]                  | AsJagged(AsDtype('>f4'))
    Jet_Pz               | float[]                  | AsJagged(AsDtype('>f4'))
    Jet_E                | float[]                  | AsJagged(AsDtype('>f4'))
    Jet_btag             | float[]                  | AsJagged(AsDtype('>f4'))
    Jet_ID               | bool[]                   | AsJagged(AsDtype('bool'))
    NMuon                | int32_t                  | AsDtype('>i4')
    Muon_Px              | float[]                  | AsJagged(AsDtype('>f4'))
    Muon_Py              | float[]                  | AsJagged(AsDtype('>f4'))
    Muon_Pz              | float[]                  | AsJagged(AsDtype('>f4'))
    Muon_E               | float[]                  | AsJagged(AsDtype('>f4'))
    Muon_Charge          | int32_t[]                | AsJagged(AsDtype('>i4'))
    Muon_Iso             | float[]                  | AsJagged(AsDtype('>f4'))

These datasets have a natural expression as Awkward Arrays:

.. code-block:: python

    >>> events.keys(filter_name="/(Jet|Muon)_P[xyz]/")
    ['Jet_Px', 'Jet_Py', 'Jet_Pz', 'Muon_Px', 'Muon_Py', 'Muon_Pz']
    >>> ak_arrays = events.arrays(filter_name="/(Jet|Muon)_P[xyz]/")
    >>> ak_arrays[:2].tolist()
    [{'Jet_Px': [],
      'Jet_Py': [],
      'Jet_Pz': [],
      'Muon_Px': [-52.89945602416992, 37.7377815246582],
      'Muon_Py': [-11.654671669006348, 0.6934735774993896],
      'Muon_Pz': [-8.16079330444336, -11.307581901550293]},
     {'Jet_Px': [-38.87471389770508],
      'Jet_Py': [19.863452911376953],
      'Jet_Pz': [-0.8949416279792786],
      'Muon_Px': [-0.8164593577384949],
      'Muon_Py': [-24.404258728027344],
      'Muon_Pz': [20.199968338012695]}]

See the `Awkward Array documentation <https://awkward-array.org>`__ for data analysis techniques using these types. (Python for loops work, but it's faster and usually more convenient to use Awkward Array's suite of NumPy-like functions.)

The same dataset *can* be read as a NumPy array with ``dtype="O"`` (Python objects), which puts NumPy arrays inside of NumPy arrays.

.. code-block:: python

    >>> np_arrays = events.arrays(filter_name="/(Jet|Muon)_P[xyz]/", library="np")
    >>> np_arrays
    {'Jet_Px': array([array([], dtype=float32), array([-38.874714], dtype=float32),
           array([], dtype=float32), ..., array([-3.7148185], dtype=float32),
           array([-36.361286, -15.256871], dtype=float32),
           array([], dtype=float32)], dtype=object),
     'Jet_Py': array([array([], dtype=float32), array([19.863453], dtype=float32),
           array([], dtype=float32), ..., array([-37.202377], dtype=float32),
           array([ 10.173571, -27.175364], dtype=float32),
           array([], dtype=float32)], dtype=object),
     'Jet_Pz': array([array([], dtype=float32), array([-0.8949416], dtype=float32),
           array([], dtype=float32), ..., array([41.012222], dtype=float32),
           array([226.42921 ,  12.119683], dtype=float32),
           array([], dtype=float32)], dtype=object),
     'Muon_Px': array([array([-52.899456,  37.73778 ], dtype=float32),
           array([-0.81645936], dtype=float32),
           array([48.98783  ,  0.8275667], dtype=float32), ...,
           array([-29.756786], dtype=float32),
           array([1.1418698], dtype=float32),
           array([23.913206], dtype=float32)], dtype=object),
     'Muon_Py': array([array([-11.654672 ,   0.6934736], dtype=float32),
           array([-24.404259], dtype=float32),
           array([-21.723139,  29.800508], dtype=float32), ...,
           array([-15.303859], dtype=float32),
           array([63.60957], dtype=float32),
           array([-35.665077], dtype=float32)], dtype=object),
     'Muon_Pz': array([array([ -8.160793, -11.307582], dtype=float32),
           array([20.199968], dtype=float32),
           array([11.168285, 36.96519 ], dtype=float32), ...,
           array([-52.66375], dtype=float32),
           array([162.17632], dtype=float32),
           array([54.719437], dtype=float32)], dtype=object)}

These "nested" NumPy arrays are not slicable as multidimensional arrays because NumPy can't assume that all of the Python objects it contains have NumPy type.

.. code-block:: python

    >>> ak_arrays["Muon_Px"][:10, 0]    # first Muon_Px of the first 10 events
    <Array [-52.9, -0.816, 49, ... -53.2, -67] type='10 * float32'>

    >>> np_arrays["Muon_Px"][:10, 0]
    # Traceback (most recent call last):
    # File "<stdin>", line 1, in <module>
    # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

The Pandas form for this type of data is a `DataFrame with MultiIndex rows <https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html>`__.

.. code-block:: python

    >>> events.arrays(filter_name="/(Jet|Muon)_P[xyz]/", library="pd")
    (
                           Jet_Px     Jet_Py      Jet_Pz
        entry subentry                                  
        1     0        -38.874714  19.863453   -0.894942
        3     0        -71.695213  93.571579  196.296432
              1         36.606369  21.838793   91.666283
              2        -28.866419   9.320708   51.243221
        4     0          3.880162 -75.234055 -359.601624
        ...                   ...        ...         ...
        2417  0        -33.196457 -59.664749  -29.040150
              1        -26.086025 -19.068407   26.774284
        2418  0         -3.714818 -37.202377   41.012222
        2419  0        -36.361286  10.173571  226.429214
              1        -15.256871 -27.175364   12.119683

        [2773 rows x 3 columns],

                           Muon_Px    Muon_Py     Muon_Pz
        entry subentry                                  
        0     0        -52.899456 -11.654672   -8.160793
              1         37.737782   0.693474  -11.307582
        1     0         -0.816459 -24.404259   20.199968
        2     0         48.987831 -21.723139   11.168285
              1          0.827567  29.800508   36.965191
        ...                   ...        ...         ...
        2416  0        -39.285824 -14.607491   61.715790
        2417  0         35.067146 -14.150043  160.817917
        2418  0        -29.756786 -15.303859  -52.663750
        2419  0          1.141870  63.609570  162.176315
        2420  0         23.913206 -35.665077   54.719437

        [3825 rows x 3 columns]
    )

Each row of the DataFrame represents one particle and the row index is broken down into "entry" and "subentry" levels. If the selected TBranches include data with different numbers of values per entry, then the return value is not a DataFrame, but a tuple of DataFrames, one for each multiplicity. See the `Pandas documentation on joining <https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html>`__ for tips on how to analyze DataFrames with partially shared keys ("entry" but not "subentry").

Iterating over intervals of entries
-----------------------------------

If you're working with large datasets, you might not have enough memory to read all entries from the TBranches you need or you might not be able to compute derived quantities for the same number of entries.

In general, array-based workflows must iterate over batches with an optimized step size:

- If the batches are too large, you'll run out of memory.
- If the batches are too small, the process will be slowed by the overhead of preparing to calculate each batch. (Array functions like the ones in NumPy and Awkward Array do one-time setup operations in slow Python and large-scale number crunching in compiled code.)

Procedural workflows, which operate on one entry (e.g. one particle physics collision event) at a time can be seen as an extreme of the latter, in which the batch size is one.

The :py:meth:`~uproot4.behaviors.TBranch.HasBranches.iterate` method has an interface like :py:meth:`~uproot4.behaviors.TBranch.TBranch.arrays`, except that takes a ``step_size`` parameter and iterates over batches of that size, rather than returning a single array group.

.. code-block:: python

    >>> events = uproot4.open("https://scikit-hep.org/uproot/examples/Zmumu.root:events")

    >>> for batch in events.iterate(step_size=500):
    ...     print(repr(batch))
    ... 
    <Array [{Type: 'GT', Run: 148031, ... M: 87.7}] type='500 * {"Type": string, "Ru...'>
    <Array [{Type: 'GT', Run: 148031, ... M: 72.5}] type='500 * {"Type": string, "Ru...'>
    <Array [{Type: 'TT', Run: 148031, ... M: 92.9}] type='500 * {"Type": string, "Ru...'>
    <Array [{Type: 'GT', Run: 148031, ... M: 94.6}] type='500 * {"Type": string, "Ru...'>
    <Array [{Type: 'TT', Run: 148029, ... M: 96.7}] type='304 * {"Type": string, "Ru...'>

With a ``step_size`` of 500, each array group has 500 entries except the last, which can have fewer (304 in this case). Also be aware that the above example reads all TBranches! You will likely want to select TBranches (columns) and the number of entries (rows) to define a batch. (See `Filtering TBranches <#filtering-tbranches>`__ above.)

Since the optimal step size is "whatever fits in memory," it's better to tune it in memory-size units than number-of-entries units. Different data types have different numbers of bytes per item, but more importantly, different applications extract different sets of TBranches, so "*N* entries" tuned for one application would not be a good tune for another.

For this reason, it's better to set the ``step_size`` to a number of bytes, such as

.. code-block:: python

    >>> for batch in events.iterate(step_size="50 kB"):
    ...     print(repr(batch))
    ... 
    <Array [{Type: 'GT', Run: 148031, ... M: 89.6}] type='667 * {"Type": string, "Ru...'>
    <Array [{Type: 'TT', Run: 148031, ... M: 18.1}] type='667 * {"Type": string, "Ru...'>
    <Array [{Type: 'GT', Run: 148031, ... M: 94.7}] type='667 * {"Type": string, "Ru...'>
    <Array [{Type: 'GT', Run: 148029, ... M: 96.7}] type='303 * {"Type": string, "Ru...'>

(but much larger in a real case). Here, ``"50 kB"`` corresponds to 667 entries (with the last step being the remainder). It's possible to calculate the number of entries for a given memory size outside of iteration using :py:meth:`~uproot4.behaviors.TBranch.HasBranches.num_entries_for`.

.. code-block:: python

    >>> events.num_entries_for("50 kB")
    667
    >>> events.num_entries_for("50 kB", filter_name="/p[xyz][12]/")
    1530
    >>> events.keys(filter_typename="double")
    ['E1', 'px1', 'py1', 'pz1', 'pt1', 'eta1', 'phi1', 'E2', 'px2', 'py2', 'pz2', 'pt2', 'eta2',
     'phi2', 'M']
    >>> events.num_entries_for("50 kB", filter_typename="double")
    702

The number of entries for ``"50 kB"`` depends strongly on which TBranches are being requested. It's the memory size, not the number of entries, that matters most when tuning a workflow for a computer with limited memory.

See the :py:meth:`~uproot4.behaviors.TBranch.HasBranches.iterate` documentation for more, including a ``report=True`` option to get a :py:class:`~uproot4.behaviors.TBranch.Report` with each batch of data with entry numbers for bookkeeping.

.. code-block:: python

    >>> for batch, report in events.iterate(step_size="50 kB", report=True):
    ...     print(report)
    ... 
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 0, 667)
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 667, 1334)
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 1334, 2001)
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 2001, 2304)

Just as ``library="np"`` and ``library="pd"`` can be used to get NumPy and Pandas output in :py:meth:`~uproot4.behaviors.TBranch.TBranch.array` and :py:meth:`~uproot4.behaviors.TBranch.HasBranches.arrays`, it can be used to yield NumPy arrays and Pandas DataFrames iteratively:

.. code-block:: python

    >>> for batch in events.iterate(step_size="100 kB", library="pd"):
    ...     print(batch)
    ... 
         Type     Run      Event         E1  ...     eta2      phi2  Q2          M
    0      GT  148031   10507008  82.201866  ... -1.05139 -0.440873  -1  82.462692
    1      TT  148031   10507008  62.344929  ... -1.21769  2.741260   1  83.626204
    2      GT  148031   10507008  62.344929  ... -1.21769  2.741260   1  83.308465
    3      GG  148031   10507008  60.621875  ... -1.21769  2.741260   1  82.149373
    4      GT  148031  105238546  41.826389  ...  1.44434 -2.707650  -1  90.469123
    ...   ...     ...        ...        ...  ...      ...       ...  ..        ...
    1328   GT  148031  607496200   4.385337  ...  1.76576 -0.582806   1   7.039820
    1329   GT  148031  607496200   4.385337  ...  1.81014  2.523670  -1  11.655561
    1330   TT  148031  607496200   8.301393  ...  1.76576 -0.582806   1  18.127933
    1331   TT  148031  607496200   8.301393  ...  1.81014  2.523670  -1   6.952658
    1332   TT  148031  607496200   8.301393  ...  2.18148  0.343855   1   1.759080

    [1333 rows x 20 columns]
         Type     Run      Event          E1  ...      eta2      phi2  Q2          M
    1333   GT  148031  607496200    8.301393  ...  1.765760 -0.582806   1  18.099339
    1334   GT  148031  607496200    8.301393  ...  1.810140  2.523670  -1   6.959646
    1335   GG  148031  607496200  132.473942  ...  1.765760 -0.582806   1  93.373860
    1336   GT  148031  608388587   59.548441  ... -0.565288  0.529327  -1  90.782261
    1337   TT  148031  608388587   51.504863  ... -0.746182 -2.573870   1  90.685446
    ...   ...     ...        ...         ...  ...       ...       ...  ..        ...
    2299   GG  148029   99768888   32.701650  ... -0.645971 -2.404430  -1  60.047138
    2300   GT  148029   99991333  168.780121  ... -1.570440  0.037027   1  96.125376
    2301   TT  148029   99991333   81.270136  ... -1.482700 -2.775240  -1  95.965480
    2302   GT  148029   99991333   81.270136  ... -1.482700 -2.775240  -1  96.495944
    2303   GG  148029   99991333   81.566217  ... -1.482700 -2.775240  -1  96.656728

    [971 rows x 20 columns]

Iterating over many files
-------------------------

Large datasets usually consist of many files, and abstractions like `ROOT's TChain <https://root.cern.ch/doc/master/classTChain.html>`__ simplify multi-file workflows by making a collection of files look like a single file.

Uproot's :py:meth:`~uproot4.behaviors.TBranch.HasBranches.iterate` takes a step in the opposite direction: it breaks single-file access into batches, and designing a workflow around batches is like designing a workflow around files. To apply such an interface to many files, all that is needed is a way to express the list of files.

The `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__ function (as opposed to the `HasBranches.iterate <uproot4.behaviors.TBranch.HasBranches.html#iterate>`__ method) takes a list of files as its first argument:

.. code-block:: python

    >>> for batch in uproot4.iterate(["dir1/*.root:events", "dir2/*.root:events"]):
    ...     do_something...

As with the single-file method, you'll want to restrict the set of TBranches to include only those you use. (See `Filtering TBranches <#filtering-tbranches>`__ above.)

The specification of file names has to include paths to the ``TTree`` objects (more generally, :py:class:`~uproot4.behaviors.TBranch.HasBranches` objects), so the colon (``:``) separating file path and object path `described above <#finding-objects-in-a-file>` is more than just a convenience in this case. Since it is possible for file paths to include colons as part of the file or directory name, the following alternate syntax can also be used:

.. code-block:: python

    >>> for batch in uproot4.iterate([{"dir1/*.root": "events"}, {"dir2/*.root": "events"}]):
    ...     do_something...

If the ``step_size`` (same meaning as in previous section) is smaller than the file size, the last batch of each file will likely be smaller than the rest: batches from one file are not mixed with batches from another file. Thus, the largest meaningful ``step_size`` is the number of entries in the file (:py:attr:`~uproot4.behaviors.TBranch.HasBranches.num_entries`). See the next section for concatenating small files.

In multi-file iteration, the :py:class:`~uproot4.behaviors.TBranch.Report` returned by ``report=True`` distinguishes between global entry numbers (:py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_start` and :py:attr:`~uproot4.behaviors.TBranch.Report.global_entry_stop`), which start once at the beginning of iteration, and TTree entry numbers (:py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_start` and :py:attr:`~uproot4.behaviors.TBranch.Report.tree_entry_stop`), which restart at the beginning of each TTree. The :py:attr:`~uproot4.behaviors.TBranch.Report.tree`, :py:attr:`~uproot4.behaviors.TBranch.Report.file`, and :py:attr:`~uproot4.behaviors.TBranch.Report.file_path` attributes are also more useful in multi-file iteration.

Reading many files into big arrays
----------------------------------

Although it iterates over multiple files, the `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__ function is not a direct analogy of `ROOT's TChain <https://root.cern.ch/doc/master/classTChain.html>`__ because it does not make multi-file workflows look like single-file (non-iterating) workflows.

The simplest way to access many files is to concatenate them into one array. The `uproot4.concatenate <uproot4.behaviors.TBranch.concatenate.html>`__ function is a multi-file analogue of the `HasBranches.arrays <uproot4.behaviors.TBranch.HasBranches.html#arrays>`__ method, in that it returns a single array group.

.. code-block:: python

    >>> events.concatenate(["dir1/*.root:events", "dir2/*.root:events"], filter_name="p*1")
    <Array [{px1: -41.2, ... pz1: -74.8}] type='23040 * {"px1": float64, "py1": float...'>

The arrays of all files have been entirely read into memory. In general, this is only possible if

- the files are small,
- the number of files is small, or
- the selected branches do not represent a large fraction of the files.

If your computer has enough memory to do this, then it will likely be the fastest way to process the data, and it's certainly easier than accumulating partial results in a loop. However, if you're working on a small subsample that will be scaled up to a bigger analysis, then it would be a bad idea to develop your analysis with this interface. You would likely need to restructure it as a loop later.

(As a multi-file function, `uproot4.concatenate <uproot4.behaviors.TBranch.concatenate.html>`__ specifies file paths and TTree object paths just like `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__.)

Reading on demand with lazy arrays
----------------------------------

Lazy-loading is a third way to access multi-file datasets, like `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__ and `uproot4.concatenate <uproot4.behaviors.TBranch.concatenate.html>`__ above. As such, it's a third analogy with `ROOT's TChain <https://root.cern.ch/doc/master/classTChain.html>`__.

The interface to `uproot4.lazy <uproot4.behaviors.TBranch.lazy.html>`__ is like `uproot4.concatenate <uproot4.behaviors.TBranch.concatenate.html>`__ in that it returns a single object, not an iterator that you have to iterate through, but it is like `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__ in that the data are not loaded immediately and do not need to reside in memory all at once.

.. code-block:: python

    >>> array = events.lazy(["dir1/*.root:events", "dir2/*.root:events"])
    >>> array
    <Array [{Type: 'GT', Run: 148031, ... M: 96.7}] type='23040 * {"Type": string, "R...'>

When `uproot4.lazy <uproot4.behaviors.TBranch.lazy.html>`__ is called, it opens all of the specified files and TTree metadata, but none of the TBranch data. It uses the TBranch names and types, as well as the TTree :py:attr:`~uproot4.behaviors.TTree.TTree.num_entries`, to define the data type and prepare batches for reading. Only when you access items in the array, such as printing them to the screen or performing a calculation on them, are the relevant TBranches read (in batches).

This lazy-loading uses an Awkward Array feature, so ``library="ak"`` is the only library option.

The fact that the data are being loaded on demand is (intentionally) hidden; one of the few ways to demonstrate that it is happening is by watching its cache fill up. When we first open a lazy array, the cache is empty.

.. code-block:: python

    >>> array = uproot4.lazy("https://scikit-hep.org/uproot/examples/Zmumu.root:events",
    ...                      step_size=100)
    >>> array.cache
    <LRUArrayCache (0/100000000 bytes full) at 0x7faf787abd00>

If we then ask for a single element from a single field, it loads one TBranch-batch. Since we specified the ``step_size=100`` (much too small for a real case; the default is ``"100 MB"``), this TBranch-bath is 100 entries, or 800 bytes.

.. code-block:: python

    >>> array["px1", 0]
    -41.1952876442
    >>> array.cache
    <LRUArrayCache (800/100000000 bytes full) at 0x7faf787abd00>

Requesting another element from the same TBranch-batch doesn't load anything else. The whole batch is already in memory.

.. code-block:: python

    >>> array["px1", 1]
    35.1180497674
    >>> array.cache
    <LRUArrayCache (800/100000000 bytes full) at 0x7faf787abd00>

Requesting an element from the next TBranch-batch loads the next batch.

.. code-block:: python

    >>> array["px1", 100]
    27.3430272161
    >>> array.cache
    <LRUArrayCache (1600/100000000 bytes full) at 0x7faf787abd00>

Requesting a different TBranch also loads a batch.

.. code-block:: python

    >>> array["py1", 100]
    11.351229626
    >>> array.cache
    <LRUArrayCache (2400/100000000 bytes full) at 0x7faf787abd00>

Performing a calculation on these two fields, ``array.px1`` and ``array.py1``, loads all batches for these two TBranches. Derived quantities, such as the result of the square root operation, are normal arrays (not lazy).

.. code-block:: python

    >>> import numpy as np
    >>> np.sqrt(array.px1**2 + array.py1**2)
    <Array [44.7, 38.8, 38.8, ... 32.4, 32.4, 32.5] type='2304 * float64'>
    >>> array.cache
    <LRUArrayCache (36864/100000000 bytes full) at 0x7faf787abd00>

Although lazy arrays combine the convenience of `uproot4.concatenate <uproot4.behaviors.TBranch.concatenate.html>`__ with the gradual loading of `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__, it is not always the most efficient way to process data. Derived quantities are fully resident in memory, and most data analyses compute more quantities than they read.

Moreover, if a lazy array is larger than its cache, reading the last batches will cause the first batches to be evicted from the cache. If it is accessed again, the first batches will need to be fully re-read, which evicts the last batches, guaranteeing that data will never be found in the cache when it's needed.

For example, in a calculation like this:

.. code-block:: python

    >>> p = np.sqrt(array.px1**2 + array.py1**2 + array.pz1**2)
    >>> pt = np.sqrt(array.px1**2 + array.py1**2)

if the three TBranches ``px1``, ``py1``, ``pz1`` don't entirely fit into their shared cache or individual caches, then none of the data loaded while computing ``p`` will be available to compute ``pt``. Small enough caches can guarantee file re-reading, which would be the slowest step in simple calculations like the above.

On the other hand, if you make the cache(s) large enough to accommodate all the arrays you'll be loading, then you might as well load them entirely into memory (with `uproot4.concatenate <uproot4.behaviors.TBranch.concatenate.html>`__). Avoiding the overhead of managing lazy batch-loading can only streamline a workflow.

So when are lazy arrays useful?

Lazy arrays are especially useful for exploring a large dataset in a convenient way. If you don't know which TBranches you will be looking at, lazy arrays save you the upfront cost of reading them all, if that were even possible. You can perform calculations interactively without having to set up iterative loops, developing the pieces of a data analysis that will later be incorporated into an efficient loop based on `uproot4.iterate <uproot4.behaviors.TBranch.iterate.html>`__.

Caching and memory management
-----------------------------

HERE
