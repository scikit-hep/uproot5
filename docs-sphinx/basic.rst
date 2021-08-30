Getting started guide
=====================

Opening a file
--------------

Open a ROOT file for reading with the :doc:`uproot.reading.open` function.

.. code-block:: python

    >>> import uproot
    >>> file = uproot.open("path/to/dataset.root")

The :doc:`uproot.reading.open` function can be (and usually should be) used like this:

.. code-block:: python

    >>> with uproot.open("path/to/dataset.root") as file:
    ...     do_something...

to automatically close the file after leaving the ``with`` block. The path-name argument can be a local file (as above), a URL ("``http://``" or "``https://``"), or XRootD ("``root://``") if you have the `Python interface to XRootD <https://anaconda.org/conda-forge/xrootd>`__ installed. It can also be a Python file-like object with ``read`` and ``seek`` methods, but such objects can't be read in parallel.

The :doc:`uproot.reading.open` function has many options, including alternate handlers for each input type, ``num_workers`` to control parallel reading, and caches (``object_cache`` and ``array_cache``). The defaults attempt to optimize parallel processing, caching, and batching of remote requests, but better performance can often be obtained by tuning these parameters.

Finding objects in a file
-------------------------

The object returned by :doc:`uproot.reading.open` represents a TDirectory inside the file (``/``).

.. code-block:: python

    >>> file = uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
    >>> file
    <ReadOnlyDirectory '/' at 0x7c070dc03040>

This object is a Python `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__, which means that you can get a list of contents with :ref:`uproot.reading.ReadOnlyDirectory.keys`.

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

Data, including nested TDirectories, are not read from disk until they are explicitly requested with square brackets (or another `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__ function, like :ref:`uproot.reading.ReadOnlyDirectory.values` or :ref:`uproot.reading.ReadOnlyDirectory.items`).

You can get the names of classes without reading the objects by using :ref:`uproot.reading.ReadOnlyDirectory.classnames`.

.. code-block:: python

    >>> file.classnames()
    {'one': 'TDirectory', 'one/two': 'TDirectory', 'one/two/tree': 'TTree', 'one/tree': 'TTree',
     'three': 'TDirectory', 'three/tree': 'TTree'}

As a shortcut, you can open a file and jump straight to the object by separating the file path and object path with a colon (``:``).

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")
    >>> events
    <TTree 'events' (20 branches) at 0x78e575394b20>

Colon separators are only allowed in strings, so you can open files that have colons in their names by wrapping them in a `pathlib.Path <https://docs.python.org/3/library/pathlib.html>`__.

Extracting histograms from a file
---------------------------------

Uproot can read most types of objects, but only a few of them have been overloaded with specialized behaviors.

.. code-block:: python

    >>> file = uproot.open("https://scikit-hep.org/uproot3/examples/hepdata-example.root")
    >>> file.classnames()
    {'hpx': 'TH1F', 'hpxpy': 'TH2F', 'hprof': 'TProfile', 'ntuple': 'TNtuple'}

Classes unknown to Uproot can be accessed through their members (raw C++ members that have been serialized into the file):

.. code-block:: python

    >>> file["hpx"].all_members
    {'@fUniqueID': 0, '@fBits': 50331656, 'fName': 'hpx', 'fTitle': 'This is the px distribution',
     'fLineColor': 602, 'fLineStyle': 1, 'fLineWidth': 1, 'fFillColor': 0, 'fFillStyle': 1001,
     'fMarkerColor': 1, 'fMarkerStyle': 1, 'fMarkerSize': 1.0, 'fNcells': 102,
     'fXaxis': <TAxis (version 9) at 0x7ca18fdb83a0>,
     'fYaxis': <TAxis (version 9) at 0x7ca18fdb8940>,
     'fZaxis': <TAxis (version 9) at 0x7ca18fdb8ca0>, 'fBarOffset': 0, 'fBarWidth': 1000,
     'fEntries': 75000.0, 'fTsumw': 74994.0, 'fTsumw2': 74994.0, 'fTsumwx': -97.16475860591163,
     'fTsumwx2': 75251.86518025988, 'fMaximum': -1111.0, 'fMinimum': -1111.0, 'fNormFactor': 0.0,
     'fContour': <TArrayD [] at 0x7ca18fdb80d0>, 'fSumw2': <TArrayD [] at 0x7ca18fdb8f70>,
     'fOption': <TString '' at 0x7ca18fdbd120>, 'fFunctions': <TList of 1 items at 0x7ca18fdc30d0>,
     'fBufferSize': 0, 'fBuffer': array([], dtype=float64), 'fBinStatErrOpt': 0, 'fN': 102}

    >>> file["hpx"].member("fName")
    'hpx'

But some classes, like :doc:`uproot.behaviors.TH1.TH1`, :doc:`uproot.behaviors.TProfile.TProfile`, and :doc:`uproot.behaviors.TH2.TH2`, have high-level "behaviors" defined in :doc:`uproot.behaviors` to make them easier to use.

Histograms have :ref:`uproot.behaviors.TAxis.TAxis.edges`, :ref:`uproot.behaviors.TH1.TH1.values`, and :ref:`uproot.behaviors.TH1.TH1.errors` methods to extract histogram axes and bin contents directly into NumPy arrays. (Keep in mind that a histogram axis with *N* bins has *N + 1* edges, and that the edges include underflow and overflow as ``-np.inf`` and ``np.inf`` endpoints.)

.. code-block:: python

    >>> file["hpx"].axis().edges()
    array([ -inf, -4.  , -3.92, -3.84, -3.76, -3.68, -3.6 , -3.52, -3.44,
           -3.36, -3.28, -3.2 , -3.12, -3.04, -2.96, -2.88, -2.8 , -2.72,
           -2.64, -2.56, -2.48, -2.4 , -2.32, -2.24, -2.16, -2.08, -2.  ,
           -1.92, -1.84, -1.76, -1.68, -1.6 , -1.52, -1.44, -1.36, -1.28,
           -1.2 , -1.12, -1.04, -0.96, -0.88, -0.8 , -0.72, -0.64, -0.56,
           -0.48, -0.4 , -0.32, -0.24, -0.16, -0.08,  0.  ,  0.08,  0.16,
            0.24,  0.32,  0.4 ,  0.48,  0.56,  0.64,  0.72,  0.8 ,  0.88,
            0.96,  1.04,  1.12,  1.2 ,  1.28,  1.36,  1.44,  1.52,  1.6 ,
            1.68,  1.76,  1.84,  1.92,  2.  ,  2.08,  2.16,  2.24,  2.32,
            2.4 ,  2.48,  2.56,  2.64,  2.72,  2.8 ,  2.88,  2.96,  3.04,
            3.12,  3.2 ,  3.28,  3.36,  3.44,  3.52,  3.6 ,  3.68,  3.76,
            3.84,  3.92,  4.  ,   inf])
    >>> file["hpx"].values()
    array([2.000e+00, 2.000e+00, 3.000e+00, 1.000e+00, 1.000e+00, 2.000e+00,
           4.000e+00, 6.000e+00, 1.200e+01, 8.000e+00, 9.000e+00, 1.500e+01,
           1.500e+01, 3.100e+01, 3.500e+01, 4.000e+01, 6.400e+01, 6.400e+01,
           8.100e+01, 1.080e+02, 1.240e+02, 1.560e+02, 1.650e+02, 2.090e+02,
           2.620e+02, 2.970e+02, 3.920e+02, 4.320e+02, 4.660e+02, 5.210e+02,
           6.040e+02, 6.570e+02, 7.880e+02, 9.030e+02, 1.079e+03, 1.135e+03,
           1.160e+03, 1.383e+03, 1.458e+03, 1.612e+03, 1.770e+03, 1.868e+03,
           1.861e+03, 1.946e+03, 2.114e+03, 2.175e+03, 2.207e+03, 2.273e+03,
           2.276e+03, 2.329e+03, 2.325e+03, 2.381e+03, 2.417e+03, 2.364e+03,
           2.284e+03, 2.188e+03, 2.164e+03, 2.130e+03, 1.940e+03, 1.859e+03,
           1.763e+03, 1.700e+03, 1.611e+03, 1.459e+03, 1.390e+03, 1.237e+03,
           1.083e+03, 1.046e+03, 8.880e+02, 7.520e+02, 7.420e+02, 6.730e+02,
           5.550e+02, 5.330e+02, 3.660e+02, 3.780e+02, 2.720e+02, 2.560e+02,
           2.000e+02, 1.740e+02, 1.320e+02, 1.180e+02, 1.000e+02, 8.900e+01,
           8.600e+01, 3.900e+01, 3.700e+01, 2.500e+01, 2.300e+01, 2.000e+01,
           1.600e+01, 1.400e+01, 9.000e+00, 1.300e+01, 8.000e+00, 2.000e+00,
           2.000e+00, 6.000e+00, 1.000e+00, 0.000e+00, 1.000e+00, 4.000e+00],
          dtype=float32)
    >>> file["hprof"].errors()
    array([0.24254264, 0.74212103, 0.49400663, 0.        , 0.        ,
          0.24649804, 0.55553737, 0.24357922, 0.22461613, 0.34906168,
          0.43563347, 0.51286511, 0.20863074, 0.28308077, 0.28915414,
          0.16769727, 0.17257732, 0.12765099, 0.10176558, 0.15209837,
          0.11509671, 0.1014912 , 0.1143207 , 0.09759737, 0.09257268,
          0.06761853, 0.07883833, 0.06391972, 0.07016808, 0.06790635,
          0.05330255, 0.05630489, 0.05523831, 0.04797496, 0.04255815,
          0.04422412, 0.04089869, 0.03453675, 0.03943858, 0.03461427,
          0.03618794, 0.03408547, 0.03170797, 0.03121938, 0.03011256,
          0.02926609, 0.03012814, 0.02977365, 0.02974839, 0.03081958,
          0.0313295 , 0.0293942 , 0.02925847, 0.0293043 , 0.02804402,
          0.03117598, 0.03010833, 0.03149117, 0.02909491, 0.0325676 ,
          0.03445547, 0.03480207, 0.0327122 , 0.03860859, 0.03885261,
          0.03856341, 0.04624045, 0.04543318, 0.04864621, 0.05203739,
          0.04324402, 0.05850656, 0.05970975, 0.0659423 , 0.07220151,
          0.08170132, 0.08712811, 0.08092333, 0.09191357, 0.10837656,
          0.10509033, 0.15493381, 0.12013956, 0.11435862, 0.183943  ,
          0.36368702, 0.13346263, 0.18325723, 0.17988976, 0.19265302,
          0.35247309, 0.18420323, 0.59593532, 0.21540243, 0.11755951,
          1.66198443, 0.13528127, 0.45343914, 0.        , 0.        ,
          0.        , 0.1681792 ])

Since Uproot is an I/O library, it intentionally does not have methods for plotting or manipulating histograms. Instead, it has methods for exporting them to other libraries.

.. code-block:: python

    >>> file["hpxpy"].to_numpy()
    (array([[0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           ...,
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.],
           [0., 0., 0., ..., 0., 0., 0.]], dtype=float32),
     array([-4. , -3.8, -3.6, -3.4, -3.2, -3. , -2.8, -2.6, -2.4, -2.2, -2. ,
           -1.8, -1.6, -1.4, -1.2, -1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,
            0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8,  2. ,  2.2,  2.4,
            2.6,  2.8,  3. ,  3.2,  3.4,  3.6,  3.8,  4. ]),
     array([-4. , -3.8, -3.6, -3.4, -3.2, -3. , -2.8, -2.6, -2.4, -2.2, -2. ,
           -1.8, -1.6, -1.4, -1.2, -1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,
            0.4,  0.6,  0.8,  1. ,  1.2,  1.4,  1.6,  1.8,  2. ,  2.2,  2.4,
            2.6,  2.8,  3. ,  3.2,  3.4,  3.6,  3.8,  4. ]))

    >>> file["hpxpy"].to_boost()
    Histogram(
      Regular(40, -4, 4),
      Regular(40, -4, 4),
      storage=Double()) # Sum: 74985.0 (75000.0 with flow)

    >>> file["hpxpy"].to_hist()
    # Traceback (most recent call last):
    #   File "/home/jpivarski/irishep/uproot/uproot/extras.py", line 237, in hist
    #     import hist
    # ModuleNotFoundError: No module named 'hist'
    #
    # During handling of the above exception, another exception occurred:
    #
    # Traceback (most recent call last):
    #   File "<stdin>", line 1, in <module>
    #   File "/home/jpivarski/irishep/uproot/uproot/behaviors/TH2.py", line 127, in to_hist
    #     return uproot.extras.hist().Hist(self.to_boost())
    #   File "/home/jpivarski/irishep/uproot/uproot/extras.py", line 239, in hist
    #     raise ImportError(
    # ImportError: install the 'hist' package with:
    #
    #     pip install hist

If one of those libraries is not currently installed, a hint is provided for how to get it.

After installing hist, we see

.. code-block:: python

    >>> file["hpxpy"].to_hist()
    Hist(
      Regular(40, -4, 4, name='xaxis', label='xaxis'),
      Regular(40, -4, 4, name='yaxis', label='yaxis'),
      storage=Double()) # Sum: 74985.0 (75000.0 with flow)

For histogramming, I recommend

- `mplhep <https://github.com/scikit-hep/mplhep>`__ for plotting NumPy-like histograms in Matplotlib.
- `boost-histogram <https://boost-histogram.readthedocs.io/>`__ for fast filling and manipulation.
- `hist <https://hist.readthedocs.io/>`__ for plotting, filling, manipulation, and fitting all in one package.

Inspecting a TBranches of a TTree
---------------------------------

:doc:`uproot.behaviors.TTree.TTree`, with the lists of :doc:`uproot.behaviors.TBranch.TBranch` it contains, are Uproot's most important "overloaded behaviors." Like :doc:`uproot.reading.ReadOnlyDirectory`, a TTree is a `Mapping <https://docs.python.org/3/library/stdtypes.html#mapping-types-dict>`__, though it maps TBranch names to the (already read) :doc:`uproot.behaviors.TBranch.TBranch` objects it contains. Since TBranches can contain more TBranches, both of these are subclasses of a general :doc:`uproot.behaviors.TBranch.HasBranches`.

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")

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

Like a TDirectory's :ref:`uproot.reading.ReadOnlyDirectory.classnames`, you can access the TBranch data types without reading data by calling :ref:`uproot.behaviors.TBranch.HasBranches.typenames`.

.. code-block:: python

    >>> events.typenames()
    {'Type': 'char*', 'Run': 'int32_t', 'Event': 'int32_t', 'E1': 'double', 'px1': 'double',
     'py1': 'double', 'pz1': 'double', 'pt1': 'double', 'eta1': 'double', 'phi1': 'double',
     'Q1': 'int32_t', 'E2': 'double', 'px2': 'double', 'py2': 'double', 'pz2': 'double',
     'pt2': 'double', 'eta2': 'double', 'phi2': 'double', 'Q2': 'int32_t', 'M': 'double'}

In an interactive session, it's often more convenient to call :ref:`uproot.behaviors.TBranch.HasBranches.show`.

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

A TBranch may be turned into an array with the :ref:`uproot.behaviors.TBranch.TBranch.array` method. The array is not read from disk until this method is called (or other array-fetching methods described below).

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")
    >>> events["M"].array()
    <Array [82.5, 83.6, 83.3, ... 96, 96.5, 96.7] type='2304 * float64'>

By default, the array is an Awkward Array, as shown above. This assumes that Awkward Array is installed (see `How to install <index.html#how-to-install>`__). If you can't install it or want to use NumPy for other reasons, pass ``library="np"`` instead of the default ``library="ak"`` or globally set ``uproot.default_library``.

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

If you don't have the specified library (including the default, Awkward Array), you'll be prompted with instructions to install it.

.. code-block:: python

    >>> events["M"].array(library="pd")
    Traceback (most recent call last):
      File "/home/jpivarski/irishep/uproot/uproot/extras.py", line 43, in pandas
        import pandas
    ModuleNotFoundError: No module named 'pandas'

    ...

    ImportError: install the 'pandas' package with:

        pip install pandas

    or

        conda install pandas

The :ref:`uproot.behaviors.TBranch.TBranch.array` method has many options, including limitations on reading (``entry_start`` and ``entry_stop``), parallelization (``decompression_executor`` and ``interpretation_executor``), and caching (``array_cache``). For details, see the reference documentation for :ref:`uproot.behaviors.TBranch.TBranch.array`.

Reading multiple TBranches as a group of arrays
-----------------------------------------------

To read more than one TBranch, you could use the :ref:`uproot.behaviors.TBranch.TBranch.array` method from the previous section multiple times, but you could also use :ref:`uproot.behaviors.TBranch.HasBranches.arrays` (plural) on the TTree itself.

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")

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

If no arguments are passed to :ref:`uproot.behaviors.TBranch.HasBranches.arrays`, *all* TBranches will be read. If your file has many TBranches, this might not be desirable or possible. You can select specific TBranches by name, as in the previous section, but you can also use a filter (``filter_name``, ``filter_typename``, or ``filter_branch``) to select TBranches by name, type, or other attributes.

The :ref:`uproot.behaviors.TBranch.HasBranches.keys`, :ref:`uproot.behaviors.TBranch.HasBranches.values`, :ref:`uproot.behaviors.TBranch.HasBranches.items`, and :ref:`uproot.behaviors.TBranch.HasBranches.typenames` methods take the same arguments, so you can test your filters before reading any data.

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")

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

The first argument of :ref:`uproot.behaviors.TBranch.HasBranches.arrays`, which we used above to pass explicit TBranch names,

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")

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

or with additional cut conditions expressed using parentheses, the cut array (below) has 269 entries.

.. code-block:: python

    >>> events.arrays(["M"], "(pt1 > 50) & ((E1>100) | (E1<90))", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{M: 91.8}, {M: 91.9, ... {M: 96.1}] type='269 * {"M": float64}'>


Note that expressions are *not*, in general, computed more quickly if expressed in these strings. The above is equivalent to the following:

.. code-block:: python

    >>> import numpy as np
    >>> arrays = events.arrays(["px1", "py1", "M"])
    >>> pt1 = np.sqrt(arrays.px1**2 + arrays.py1**2)
    >>> arrays.M[pt1 > 50]
    <Array [91.8, 91.9, 91.7, ... 90.1, 90.1, 96.1] type='289 * float64'>

but perhaps more convenient. If what you want to compute requires more than one expression, you'll have to move it out of strings into Python.

The default ``language`` is :doc:`uproot.language.python.PythonLanguage`, but other languages, like ROOT's `TTree::Draw syntax <https://root.cern.ch/doc/master/classTTree.html#a73450649dc6e54b5b94516c468523e45>`_ are foreseen *in the future*. Thus, implicit loops (e.g. ``Sum$(...)``) have to be translated to their Awkward equivalents and ``ROOT::Math`` functions have to be translated to their NumPy equivalents.

Nested data structures
----------------------

Not all datasets have one value per entry. In particle physics, we often have different numbers of particles (and particle attributes) per collision event.

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/HZZ.root:events")
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

The :ref:`uproot.behaviors.TBranch.HasBranches.iterate` method has an interface like :ref:`uproot.behaviors.TBranch.TBranch.arrays`, except that takes a ``step_size`` parameter and iterates over batches of that size, rather than returning a single array group.

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")

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

(but much larger in a real case). Here, ``"50 kB"`` corresponds to 667 entries (with the last step being the remainder). It's possible to calculate the number of entries for a given memory size outside of iteration using :ref:`uproot.behaviors.TBranch.HasBranches.num_entries_for`.

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

See the :ref:`uproot.behaviors.TBranch.HasBranches.iterate` documentation for more, including a ``report=True`` option to get a :doc:`uproot.behaviors.TBranch.Report` with each batch of data with entry numbers for bookkeeping.

.. code-block:: python

    >>> for batch, report in events.iterate(step_size="50 kB", report=True):
    ...     print(report)
    ...
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 0, 667)
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 667, 1334)
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 1334, 2001)
    Report(<TTree 'events' (20 branches) at 0x7e8391770310>, 2001, 2304)

Just as ``library="np"`` and ``library="pd"`` can be used to get NumPy and Pandas output in :ref:`uproot.behaviors.TBranch.TBranch.array` and :ref:`uproot.behaviors.TBranch.HasBranches.arrays`, it can be used to yield NumPy arrays and Pandas DataFrames iteratively:

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

Uproot's :ref:`uproot.behaviors.TBranch.HasBranches.iterate` takes a step in the opposite direction: it breaks single-file access into batches, and designing a workflow around batches is like designing a workflow around files. To apply such an interface to many files, all that is needed is a way to express the list of files.

The :doc:`uproot.behaviors.TBranch.iterate` function (as opposed to the :ref:`uproot.behaviors.TBranch.HasBranches.iterate` method) takes a list of files as its first argument:

.. code-block:: python

    >>> for batch in uproot.iterate(["dir1/*.root:events", "dir2/*.root:events"]):
    ...     do_something...

As with the single-file method, you'll want to restrict the set of TBranches to include only those you use. (See `Filtering TBranches <#filtering-tbranches>`__ above.)

The specification of file names has to include paths to the ``TTree`` objects (more generally, :doc:`uproot.behaviors.TBranch.HasBranches` objects), so the colon (``:``) separating file path and object path `described above <#finding-objects-in-a-file>` is more than just a convenience in this case. Since it is possible for file paths to include colons as part of the file or directory name, the following alternate syntax can also be used:

.. code-block:: python

    >>> for batch in uproot.iterate([{"dir1/*.root": "events"}, {"dir2/*.root": "events"}]):
    ...     do_something...

If the ``step_size`` (same meaning as in previous section) is smaller than the file size, the last batch of each file will likely be smaller than the rest: batches from one file are not mixed with batches from another file. Thus, the largest meaningful ``step_size`` is the number of entries in the TTree (:ref:`uproot.behaviors.TTree.TTree.num_entries`). See the next section for concatenating small files.

In multi-file iteration, the :doc:`uproot.behaviors.TBranch.Report` returned by ``report=True`` distinguishes between global entry numbers (:ref:`uproot.behaviors.TBranch.Report.global_entry_start` and :ref:`uproot.behaviors.TBranch.Report.global_entry_stop`), which start once at the beginning of iteration, and TTree entry numbers (:ref:`uproot.behaviors.TBranch.Report.tree_entry_start` and :ref:`uproot.behaviors.TBranch.Report.tree_entry_stop`), which restart at the beginning of each TTree. The :ref:`uproot.behaviors.TBranch.Report.tree`, :ref:`uproot.behaviors.TBranch.Report.file`, and :ref:`uproot.behaviors.TBranch.Report.file_path` attributes are also more useful in multi-file iteration.

Reading many files into big arrays
----------------------------------

Although it iterates over multiple files, the :doc:`uproot.behaviors.TBranch.iterate` function is not a direct analogy of `ROOT's TChain <https://root.cern.ch/doc/master/classTChain.html>`__ because it does not make multi-file workflows look like single-file (non-iterating) workflows.

The simplest way to access many files is to concatenate them into one array. The :doc:`uproot.behaviors.TBranch.concatenate` function is a multi-file analogue of the :ref:`uproot.behaviors.TBranch.HasBranches.arrays` method, in that it returns a single array group.

.. code-block:: python

    >>> uproot.concatenate(["dir1/*.root:events", "dir2/*.root:events"], filter_name="p*1")
    <Array [{px1: -41.2, ... pz1: -74.8}] type='23040 * {"px1": float64, "py1": float...'>

The arrays of all files have been entirely read into memory. In general, this is only possible if

- the files are small,
- the number of files is small, or
- the selected branches do not represent a large fraction of the files.

If your computer has enough memory to do this, then it will likely be the fastest way to process the data, and it's certainly easier than accumulating partial results in a loop. However, if you're working on a small subsample that will be scaled up to a bigger analysis, then it would be a bad idea to develop your analysis with this interface. You would likely need to restructure it as a loop later.

(As a multi-file function, :doc:`uproot.behaviors.TBranch.concatenate` specifies file paths and TTree object paths just like :doc:`uproot.behaviors.TBranch.iterate`.)

Reading on demand with lazy arrays
----------------------------------

Lazy-loading is a third way to access multi-file datasets, like :doc:`uproot.behaviors.TBranch.iterate` and :doc:`uproot.behaviors.TBranch.concatenate` above. As such, it's a third analogy with `ROOT's TChain <https://root.cern.ch/doc/master/classTChain.html>`__.

The interface to :doc:`uproot.behaviors.TBranch.lazy` is like :doc:`uproot.behaviors.TBranch.concatenate` in that it returns a single object, not an iterator that you have to iterate through, but it is like :doc:`uproot.behaviors.TBranch.iterate` in that the data are not loaded immediately and do not need to reside in memory all at once.

.. code-block:: python

    >>> array = uproot.lazy(["dir1/*.root:events", "dir2/*.root:events"])
    >>> array
    <Array [{Type: 'GT', Run: 148031, ... M: 96.7}] type='23040 * {"Type": string, "R...'>

When :doc:`uproot.behaviors.TBranch.lazy` is called, it opens all of the specified files and TTree metadata, but none of the TBranch data. It uses the TBranch names and types, as well as the TTree :ref:`uproot.behaviors.TTree.TTree.num_entries`, to define the data type and prepare batches for reading. Only when you access items in the array, such as printing them to the screen or performing a calculation on them, are the relevant TBranches read (in batches).

This lazy-loading uses an Awkward Array feature, so ``library="ak"`` is the only library option.

The fact that the data are being loaded on demand is (intentionally) hidden; one of the few ways to demonstrate that it is happening is by watching its cache fill up. When we first open a lazy array, the cache is empty.

.. code-block:: python

    >>> cache = uproot.LRUArrayCache("1 GB")
    >>> array = uproot.lazy("https://scikit-hep.org/uproot3/examples/Zmumu.root:events",
    ...                      step_size=100,
    ...                      array_cache=cache)
    >>> cache
    <LRUArrayCache (0/100000000 bytes full) at 0x7faf787abd00>

If we then ask for a single element from a single field, it loads one TBranch-batch. Since we specified the ``step_size=100`` (much too small for a real case; the default is ``"100 MB"``), this TBranch-bath is 100 entries, or 800 bytes.

.. code-block:: python

    >>> array["px1", 0]
    -41.1952876442
    >>> cache
    <LRUArrayCache (800/100000000 bytes full) at 0x7faf787abd00>

Requesting another element from the same TBranch-batch doesn't load anything else. The whole batch is already in memory.

.. code-block:: python

    >>> array["px1", 1]
    35.1180497674
    >>> cache
    <LRUArrayCache (800/100000000 bytes full) at 0x7faf787abd00>

Requesting an element from the next TBranch-batch loads the next batch.

.. code-block:: python

    >>> array["px1", 100]
    27.3430272161
    >>> cache
    <LRUArrayCache (1600/100000000 bytes full) at 0x7faf787abd00>

Requesting a different TBranch also loads a batch.

.. code-block:: python

    >>> array["py1", 100]
    11.351229626
    >>> cache
    <LRUArrayCache (2400/100000000 bytes full) at 0x7faf787abd00>

Performing a calculation on these two fields, ``array.px1`` and ``array.py1``, loads all batches for these two TBranches. Derived quantities, such as the result of the square root operation, are normal arrays (not lazy).

.. code-block:: python

    >>> import numpy as np
    >>> np.sqrt(array.px1**2 + array.py1**2)
    <Array [44.7, 38.8, 38.8, ... 32.4, 32.4, 32.5] type='2304 * float64'>
    >>> cache
    <LRUArrayCache (36864/100000000 bytes full) at 0x7faf787abd00>

Although lazy arrays combine the convenience of :doc:`uproot.behaviors.TBranch.concatenate` with the gradual loading of :doc:`uproot.behaviors.TBranch.iterate`, it is not always the most efficient way to process data. Derived quantities are fully resident in memory, and most data analyses compute more quantities than they read.

Moreover, if a lazy array is larger than its cache, reading the last batches will cause the first batches to be evicted from the cache. If it is accessed again, the first batches will need to be fully re-read, which evicts the last batches, guaranteeing that data will never be found in the cache when it's needed.

For example, in a calculation like this:

.. code-block:: python

    >>> p = np.sqrt(array.px1**2 + array.py1**2 + array.pz1**2)
    >>> pt = np.sqrt(array.px1**2 + array.py1**2)

if the three TBranches ``px1``, ``py1``, ``pz1`` don't entirely fit into their shared cache or individual caches, then none of the data loaded while computing ``p`` will be available to compute ``pt``. Small enough caches can guarantee file re-reading, which would be the slowest step in simple calculations like the above.

On the other hand, if you make the cache(s) large enough to accommodate all the arrays you'll be loading, then you might as well load them entirely into memory (with :doc:`uproot.behaviors.TBranch.concatenate`). Avoiding the overhead of managing lazy batch-loading can only streamline a workflow.

So when are lazy arrays useful?

Lazy arrays are especially useful for exploring a large dataset in a convenient way. If you don't know which TBranches you will be looking at, lazy arrays save you the upfront cost of reading them all, if that were even possible. You can perform calculations interactively without having to set up iterative loops, developing the pieces of a data analysis that will later be incorporated into an efficient loop based on :doc:`uproot.behaviors.TBranch.iterate`.

Caching and memory management
-----------------------------

Each file has an associated ``object_cache`` and ``array_cache``, which streamline interactive use but could be surprising if you're trying to track down memory use.

The ``object_cache`` stores a number of objects like TDirectories, histograms, and TTrees. The main effect of this is that

.. code-block:: python

    >>> file = uproot.open("https://scikit-hep.org/uproot3/examples/hepdata-example.root")
    >>> histogram = file["hpx"]
    >>> (histogram, histogram)
    (<TH1F (version 1) at 0x7d9a05a43370>, <TH1F (version 1) at 0x7d9a05a43370>)

and

.. code-block:: python

    >>> (file["hpx"], file["hpx"])
    (<TH1F (version 1) at 0x7d9a05a43370>, <TH1F (version 1) at 0x7d9a05a43370>)

have identical performance. Not having to declare names for things that are already referenced by name simplifies bookkeeping.

The ``array_cache`` stores array outputs up to a maximum number of bytes. The arrays must have an ``nbytes`` or ``memory_usage`` attribute/property to track usage, which NumPy, Awkward Array, and Pandas all have. As with the ``object_cache``, the ``array_cache`` ensures that

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")
    >>> array = events["px1"].array()
    >>> (array, array)
    (<Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>,
     <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>)

and

.. code-block:: python

    >>> (events["px1"].array(), events["px1"].array())
    (<Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>,
     <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>)

have the same performance, assuming that the caches are not overrun.

By default, each file has a separate cache of ``100`` objects and ``"100 MB"`` of arrays. However, these can be overridden by passing an ``object_cache`` or ``array_cache`` argument to :doc:`uproot.reading.open` or setting the :ref:`uproot.reading.ReadOnlyFile.object_cache` and :ref:`uproot.reading.ReadOnlyFile.array_cache` properties.

Any `MutableMapping <https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes>`__ will do (including a plain dict, which would keep objects forever), or you can set them to ``None`` to prevent caching.

Parallel processing
-------------------

Data are or can be read in parallel in each of the following three stages.

- Physically reading bytes from disk or remote sources: the parallel processing or single-thread background processing is handled by the specific :doc:`uproot.source.chunk.Source` type, which can be influenced with :doc:`uproot.reading.open` options (particularly ``num_workers`` and ``num_fallback_workers``).
- Decompressing TBasket (:doc:`uproot.models.TBasket.Model_TBasket`) data: depends on the ``decompression_executor``.
- Interpreting decompressed data with an array :doc:`uproot.interpretation.Interpretation`: depends on the ``interpretation_executor``.

Like the caches, the default values for the last two are global ``uproot.decompression_executor`` and ``uproot.interpretation_executor`` objects. The default ``decompression_executor`` is a :doc:`uproot.source.futures.ThreadPoolExecutor` with as many workers as your computer has CPU cores. Decompression workloads are executed in compiled extensions with the `Python GIL <https://wiki.python.org/moin/GlobalInterpreterLock>`__ released, so they can afford to run with full parallelism. The default ``interpretation_executor`` is a :doc:`uproot.source.futures.TrivialExecutor` that behaves like an distributed executor, but actually runs sequentially. Most interpretation workflows are not computationally intensive or are currently implemented in Python, so they would not currently benefit from parallelism.

If, however, you're working in an environment that puts limits on parallel processing (e.g. the CMS LPC or informal university computers), you may want to modify the defaults, either locally through a ``decompression_executor`` or ``interpretation_executor`` function parameter, or globally by replacing the global object.

Opening a file for writing
--------------------------

All of the above describes reading data only. If you want to *write* to ROOT files, you open them in a different way:

.. code-block:: python

    >>> file = uproot.recreate("path/to/new-file.root")

or

.. code-block:: python

    >>> file = uproot.update("path/to/existing-file.root")

The :doc:`uproot.writing.writable.recreate` function creates a new file, deleting any that might have previously existed with that name, and :doc:`uproot.writing.writable.update` opens a preexisting file to add to it or delete some of its objects. These correspond to ``"RECREATE"`` and ``"UPDATE"`` in ROOT (as well as the less often used :doc:`uproot.writing.writable.create` for ``"CREATE"``).

All of these functions can be (and usually should be) used like this:

.. code-block:: python

    >>> with uproot.recreate("/path/to/new-file.root") as file:
    ...     do_something...

to automatically close the file after leaving the ``with`` block.

The key thing to be aware of is that writing is completely separate from reading: these functions return a :doc:`uproot.writing.writable.WritableDirectory`, rather than the :doc:`uproot.reading.ReadOnlyDirectory` that :doc:`uproot.reading.open` returns, and these objects have different methods.

Writing objects to a file
-------------------------

The object returned by :doc:`uproot.writing.writable.recreate` or :doc:`uproot.writing.writable.update` represents a TDirectory inside the file.

.. code-block:: python

    >>> file = uproot.recreate("example.root")
    >>> file
    <WritableDirectory '/' at 0x7fad19df3cd0>

This object is a Python `MutableMapping <https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping>`__, which means that you can add data to it by assignment.

.. code-block:: python

    >>> import numpy as np
    >>> file["hist"] = np.histogram(np.random.normal(0, 1, 100000))
    >>> file["hist"]
    <TH1D (version 3) at 0x7fad19e0a550>

To put data in a nested directory, just include slashes in the name.

.. code-block:: python

    >>> file["subdir/hist"] = np.histogram(np.random.normal(0, 1, 100000))
    >>> file["subdir/hist"]
    <TH1D (version 3) at 0x7fad1d472e20>

    >>> file["subdir/README"] = "This directory has all the stuff in it."
    >>> file["subdir/README"]
    <TObjString 'This directory has all the stuff in it.' at 0x7faca9c354a0>
    >>> file.keys()
    ['hist;1', 'subdir;1', 'subdir/hist;1', 'subdir/README;1']
    >>> file.classnames()
    {'hist;1': 'TH1D',
     'subdir;1': 'TDirectory',
     'subdir/hist;1': 'TH1D',
     'subdir/README;1': 'TObjString'}

Empty directories can be made with the :ref:`uproot.writing.writable.WritableDirectory.mkdir` method.

.. note::

    A small but growing list of data types can be written to files:

    * strings: TObjString
    * histograms: TH1*, TH2*, TH3*
    * profile plots: TProfile, TProfile2D, TProfile3D
    * NumPy histograms created with `np.histogram <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__, `np.histogram2d <https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html>`__, and `np.histogramdd <https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html>`__ with 3 dimensions or fewer
    * histograms that satisfy the `Universal Histogram Interface <https://uhi.readthedocs.io/>`__ (UHI) with 3 dimensions or fewer; this includes `boost-histogram <https://boost-histogram.readthedocs.io/>`__ and `hist <https://hist.readthedocs.io/>`__
    * PyROOT objects

Here is an example using hist:

.. code-block:: python

    >>> import hist
    >>> h = hist.Hist.new.Reg(10, -5, 5, name="x").Weight()
    >>> h.fill(np.random.normal(0, 1, 100000))
    Hist(Regular(10, -5, 5, name='x', label='x'), storage=Weight()) # Sum: WeightedSum(value=100000, variance=100000)
    >>> file["from_hist"] = h
    >>> file["from_hist"]
    <TH1D (version 3) at 0x7f5fb6e78970>

And here's an example using PyROOT:

.. code-block:: python

    >>> import ROOT
    >>> pyroot_hist = ROOT.TH1F("h", "", 100, -3, 3)
    >>> pyroot_hist.FillRandom("gaus", 100000)
    >>> file["from_pyroot"] = pyroot_hist
    >>> file["from_pyroot"]
    <TH1F (version 3) at 0x7facaa8aac10>

This makes use of the :doc:`uproot.pyroot.from_pyroot` function, which turns any (readable) PyROOT object into its corresponding :doc:`uproot.model.Model`.

.. code-block:: python

    >>> uproot.from_pyroot(pyroot_hist)
    <TH1F (version 3) at 0x7facaa8b6df0>
    >>> uproot.from_pyroot(pyroot_hist).to_numpy()
    (array([  28.,   24.,   36.,   50.,   70.,   71.,   86.,  101.,   82.,
             128.,  139.,  181.,  187.,  218.,  251.,  281.,  345.,  355.,
             387.,  482.,  492.,  557.,  577.,  691.,  701.,  820.,  919.,
             882., 1016., 1122., 1269., 1353., 1426., 1474., 1517., 1610.,
            1700., 1818., 1844., 2002., 2070., 2195., 2219., 2177., 2272.,
            2278., 2347., 2407., 2431., 2410., 2407., 2462., 2375., 2388.,
            2284., 2274., 2235., 2209., 2138., 1996., 1895., 1800., 1789.,
            1698., 1648., 1604., 1478., 1399., 1264., 1213., 1128., 1019.,
             948.,  861.,  825.,  739.,  636.,  631.,  511.,  499.,  464.,
             420.,  384.,  296.,  314.,  258.,  235.,  187.,  159.,  134.,
             121.,  101.,   92.,   78.,   79.,   63.,   49.,   38.,   42.,
              35.], dtype=float32),
     array([-3.  , -2.94, -2.88, -2.82, -2.76, -2.7 , -2.64, -2.58, -2.52,
            -2.46, -2.4 , -2.34, -2.28, -2.22, -2.16, -2.1 , -2.04, -1.98,
            -1.92, -1.86, -1.8 , -1.74, -1.68, -1.62, -1.56, -1.5 , -1.44,
            -1.38, -1.32, -1.26, -1.2 , -1.14, -1.08, -1.02, -0.96, -0.9 ,
            -0.84, -0.78, -0.72, -0.66, -0.6 , -0.54, -0.48, -0.42, -0.36,
            -0.3 , -0.24, -0.18, -0.12, -0.06,  0.  ,  0.06,  0.12,  0.18,
             0.24,  0.3 ,  0.36,  0.42,  0.48,  0.54,  0.6 ,  0.66,  0.72,
             0.78,  0.84,  0.9 ,  0.96,  1.02,  1.08,  1.14,  1.2 ,  1.26,
             1.32,  1.38,  1.44,  1.5 ,  1.56,  1.62,  1.68,  1.74,  1.8 ,
             1.86,  1.92,  1.98,  2.04,  2.1 ,  2.16,  2.22,  2.28,  2.34,
             2.4 ,  2.46,  2.52,  2.58,  2.64,  2.7 ,  2.76,  2.82,  2.88,
             2.94,  3.  ]))

Removing objects from a file
----------------------------

As usual with a `MutableMapping <https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping>`__, you can delete objects with the ``del`` operator.

.. code-block:: python

    >>> file.keys()
    ['hist;1', 'subdir;1', 'subdir/hist;1', 'subdir/README;1', 'from_hist;1', 'from_pyroot;1']
    >>> del file["from_pyroot"]
    >>> del file["from_hist"]
    >>> del file["hist"]
    >>> file.keys()
    ['subdir;1', 'subdir/hist;1', 'subdir/README;1']

This can delete objects created by Uproot or objects created by ROOT if the file was opened with :doc:`uproot.writing.writable.update`.

Writing TTrees to a file
------------------------

TTrees are a special type of object, just as TDirectories are special: data can be cumulatively added to them.

However, :doc:`uproot.writing.writable.WritableTree` objects can be created in the same way as static objects, by assigning TTree-like data to a name in a directory.

.. code-block:: python

    >>> file["tree1"] = {"branch1": np.arange(1000), "branch2": np.arange(1000)*1.1}
    >>> file["tree1"]
    <WritableTree '/tree1' at 0x7f2ede193e20>
    >>> file["tree1"].show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    branch1              | int64_t                  | AsDtype('>i8')
    branch2              | double                   | AsDtype('>f8')

Python dicts of equal-length NumPy arrays are TTree-like, as are Pandas DataFrames:

.. code-block:: python

    >>> import pandas as pd
    >>> df = pd.DataFrame({"x": np.arange(1000), "y": np.arange(1000)*1.1})
    >>> df
           x       y
    0      0     0.0
    1      1     1.1
    2      2     2.2
    3      3     3.3
    4      4     4.4
    ..   ...     ...
    995  995  1094.5
    996  996  1095.6
    997  997  1096.7
    998  998  1097.8
    999  999  1098.9

    [1000 rows x 2 columns]
    >>> file["tree2"] = df
    >>> file["tree2"]
    <WritableTree '/tree2' at 0x7f2e7c516d90>
    >>> file["tree2"].show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    index                | int64_t                  | AsDtype('>i8')
    x                    | int64_t                  | AsDtype('>i8')
    y                    | double                   | AsDtype('>f8')

If the arrays are Awkward Arrays, they can contain a variable number of values per entry:

.. code-block:: python

    >>> import awkward as ak
    >>> file["tree3"] = {"branch": ak.Array([[1.1, 2.2, 3.3], [], [4.4, 5.5]])}
    >>> file["tree3"]
    <WritableTree '/tree3' at 0x7f2e7c516dc0>
    >>> file["tree3"].show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    nbranch              | int32_t                  | AsDtype('>i4')
    branch               | double[]                 | AsJagged(AsDtype('>f8'))

And Awkward record arrays, constructed with `ak.zip <https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html>`__, can consolidate arrays to ensure that there is only one "counter" TBranch.

.. code-block:: python

    >>> file["tree4"] = {"Muon": ak.zip({"pt": muon_pt, "eta": muon_eta, "phi": muon_phi})}
    >>> file["tree4"]
    <WritableTree '/tree4' at 0x7fee9e3ebc40>
    >>> file["tree4"].show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    nMuon                | int32_t                  | AsDtype('>i4')
    Muon_pt              | double[]                 | AsJagged(AsDtype('>f8'))
    Muon_eta             | double[]                 | AsJagged(AsDtype('>f8'))
    Muon_phi             | double[]                 | AsJagged(AsDtype('>f8'))

.. note::

    The small but growing list of data types can be written as TTrees is:

    * dict of NumPy arrays (flat, multidimensional, and/or structured), Awkward Arrays containing one level of variable-length lists and/or one level of records, or a Pandas DataFrame with a numeric index
    * a single NumPy structured array (one level deep)
    * a single Awkward Array containing one level of variable-length lists and/or one level of records
    * a single Pandas DataFrame with a numeric index

Just as empty directories can be made with the :ref:`uproot.writing.writable.WritableDirectory.mkdir` method, empty TTrees can be made with :ref:`uproot.writing.writable.WritableDirectory.mktree`.

.. code-block:: python

    >>> file.mktree("tree5", {"x": ("f4", (3,)), "y": "var * int64"}, title="A title")
    <WritableTree '/tree5' at 0x7fee9d3a5190>
    >>> file["tree5"].show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    x                    | float[3]                 | AsDtype("('>f4', (3,))")
    ny                   | int32_t                  | AsDtype('>i4')
    y                    | int64_t[]                | AsJagged(AsDtype('>i8'))

This method also provides control over the naming convention for counter TBranches and subfield TBranches (for structured NumPy, Pandas DataFrames, and Awkward record arrays inside a dict); see its documentation.

Extending TTrees with large datasets
------------------------------------

It's likely that you'll want to write more data to disk than can fit in memory. The data in a :doc:`uproot.writing.writable.WritableTree` can be extended with the :ref:`uproot.writing.writable.WritableTree.extend` method (named in analogy with Python's `list.extend <https://docs.python.org/3/tutorial/datastructures.html#more-on-lists>`__).

Using ``"tree5"`` as an example (above),

.. code-block:: python

    >>> file["tree5"].num_entries, file["tree5"].num_baskets
    (0, 0)

    >>> file["tree5"].extend({
    ...     "x": np.arange(15).reshape(5, 3),
    ...     "y": ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
    ... })
    >>> file["tree5"].num_entries, file["tree5"].num_baskets
    (5, 1)

    >>> file["tree5"].extend({
    ...     "x": np.arange(15).reshape(5, 3),
    ...     "y": ak.Array([[0.0, 1.1, 2.2], [], [3.3, 4.4], [5.5], [6.6, 7.7, 8.8, 9.9]])
    ... })
    >>> file["tree5"].num_entries, file["tree5"].num_baskets
    (10, 2)

The :ref:`uproot.writing.writable.WritableTree.extend` method always adds one TBasket to each TBranch in the TTree. The data you provide must have the types that have been established in the first write or :ref:`uproot.writing.writable.WritableDirectory.mktree` call: exactly the same set of TBranch names and the same data type for each TBranch (or castable to it).

The arrays also have to have the same lengths as each other, though only in the first dimension. Above, the ``"x"`` NumPy array has shape ``(5, 3)``: the first dimension has length 5. The ``"y"`` Awkward array has type ``5 * var * float64``: the first dimension has length 5. This is why they are compatible; the inner dimensions don't matter (except inasmuch as they have the right *type*).

.. warning::

    **As a word of warning,** be sure that each call to :ref:`uproot.writing.writable.WritableTree.extend` includes at least 100 kB per branch/array. (NumPy and Awkward Arrays have an `nbytes <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.nbytes.html>`__ property; you want at least ``100000`` per array.) If you ask Uproot to write very small TBaskets, such as the examples with length ``5`` above, it will spend more time working on TBasket overhead than actually writing data. The absolute worst case is one-entry-per-:ref:`uproot.writing.writable.WritableTree.extend`. See `#428 (comment) <https://github.com/scikit-hep/uproot4/pull/428#issuecomment-908703486>`__.

Specifying the compression
--------------------------

You can specify the compression for a whole file while opening it:

.. code-block:: python

    >>> file = uproot.recreate("example.root", compression=uproot.ZLIB(4))
    >>> file.compression
    ZLIB(4)

This compression setting is mutable; you can change it at any time to compress some objects with one compression setting and other objects with another.

.. code-block:: python

    >>> file.compression = uproot.LZMA(9)
    >>> file.compression
    LZMA(9)

:doc:`uproot.writing.writable.WritableTree` objects also have a :ref:`uproot.writing.writable.WritableTree.compression` setting that can override the global one for the :doc:`uproot.writing.writable.WritableFile`.

.. code-block:: python

    >>> file.mktree("tree", {"x": "f4", "y": "var * int64"})
    <WritableTree '/tree' at 0x7fcaeda25640>
    >>> file["tree"].compression
    LZMA(9)
    >>> file["tree"].compression = uproot.LZ4(1)
    >>> file["tree"].compression
    LZ4(1)

In addition, each TBranch of the TTree can have a different compression setting:

.. code-block:: python

    >>> file["tree"]["x"].compression = uproot.ZSTD(1)
    >>> file["tree"]["y"].compression = uproot.ZSTD(9)
    >>> file["tree"].compression
    {'x': ZSTD(1), 'ny': LZ4(1), 'y': ZSTD(9)}
    >>> file["tree"].compression = {"x": None, "ny": None, "y": uproot.ZLIB(4)}
    >>> file["tree"].compression
    {'x': None, 'ny': None, 'y': ZLIB(4)}

Changes to the compression setting only affect TBaskets written after the change (with :ref:`uproot.writing.writable.WritableTree.extend`; see above).
