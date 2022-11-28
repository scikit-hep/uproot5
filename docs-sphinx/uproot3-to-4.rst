Uproot 3 → 4+ cheat-sheet
=========================

The Uproot 3 → 4 transition was primarily motivated by Awkward Array 0 → 1. The interface of Awkward Array significantly changed and Awkward Arrays are output by Uproot functions, so this difference would be visible to you as a user of Uproot.

Thus, it was also a good time to introduce interface changes to Uproot itself—such as presenting C++ strings as Python 3 ``str``, rather than uninterpreted ``bytes``.

Fundamental changes were also required to streamline remote reading (HTTP and XRootD), so Uproot 4 was distributed as a separate project in parallel with Uproot 3 (like Awkward 1 and 0). For the latter half of 2020, adventurous users and downstream developers could install `uproot <https://pypi.org/project/uproot/>`__ as a separate project.

.. image:: https://raw.githubusercontent.com/scikit-hep/uproot4/main/docs-img/diagrams/uproot-awkward-timeline.png
  :width: 100%

On December 1, 2020, however, Awkward 0 and Uproot 3 were deprecated, moved to PyPI packages `awkward0 <https://pypi.org/project/awkward0/>`__ and `uproot3 <https://pypi.org/project/uproot3/>`__, while Awkward 1 and Uproot 4 became unqualified as `awkward <https://pypi.org/project/awkward/>`__ and `uproot <https://pypi.org/project/uproot/>`__.

This document is to help users of Uproot 3 get started on Uproot 4 or later.

(The differences between Uproot 4 and Uproot 5 are much smaller; some functions were added, many bugs fixed, and ``uproot.lazy`` was replaced by :doc:`uproot._dask.dask`. A similar document is not needed if you're upgrading from Uproot 4 to 5, but if you're upgrading from 3 to 5, read on!)

Opening a file
--------------

The "open" function is still named :func:`uproot.open <uproot.reading.open>`, and it still recognizes local files, HTTP, and XRootD by the URL prefix (or lack thereof).

.. code-block:: python

    >>> import uproot
    >>> local_file = uproot.open("local/file.root")
    >>> http_file = uproot.open("https://server.net/file.root")
    >>> xrootd_file = uproot.open("root://server.net/file.root")

But whereas Uproot 3 took a handler (local/HTTP/XRootD) and options as a class instance, handlers are specified in Uproot 4 by passing a class and options are free-standing arguments.

.. code-block:: python

    >>> file = uproot.open("file.root",
    ...                     file_handler=uproot.MultithreadedFileSource,
    ...                     num_workers=10)
    >>> file = uproot.open("https://server.net/file.root",
    ...                     http_handler=uproot.MultithreadedHTTPSource,
    ...                     timeout=3.0)
    >>> file = uproot.open("root://server.net/file.root",
    ...                     xrootd_handler=uproot.MultithreadedXRootDSource,
    ...                     num_workers=5)

As in Uproot 3, there is a function for iterating over many files, :func:`uproot.iterate <uproot.behaviors.TBranch.iterate>`, and for lazily opening files, :func:`uproot.lazy <uproot.behaviors.TBranch.lazy>` (replacing ``uproot3.lazyarray`` and ``uproot3.lazyarrays``). Uproot 4 has an additional function for reading many files into one array (not lazily): :func:`uproot.concatenate <uproot.behaviors.TBranch.concatenate>`.

Array-reading differences are covered in `Reading arrays <#reading-arrays>`__, below. File-opening differences are illustrated well enough with :func:`uproot.open <uproot.reading.open>`.

New features
""""""""""""

Files can now truly be closed (long story), so the ``with`` syntax is recommended for scripts that open a lot of files.

.. code-block:: python

    >>> with uproot.open("file.root") as file:
    ...     do_something(file)

Python file objects can be passed to :func:`uproot.open <uproot.reading.open>` in place of a filename string.

.. code-block:: python

    >>> import tarfile
    >>> with tarfile.open("file.tar.gz", "r") as tar:
    ...     with uproot.open(tar) as file:
    ...         do_something(file)

There's a filename syntax for opening a file and pulling one object out of it. This is primarily for convenience but was strongly requested (`#79 <https://github.com/scikit-hep/uproot4/issues/79>`__).

.. code-block:: python

    >>> histogram = uproot.open("file.root:path/to/histogram")

So what if the filename has a colon (``:``) in it? (Note: URLs are properly handled.) You have two options: (1) ``pathlib.Path`` objects are never parsed for the colon separator and (2) you can also express the separation with a dict.

.. code-block:: python

    >>> histogram = uproot.open(pathlib.Path("real:colon.root"))["histogram"]
    >>> histogram = uproot.open({"real:colon.root": "histogram"})

Error messages about missing files will remind you of the options.

If you want to use this with a context manager (``with`` statement), closing the extracted object closes the file it came from.

.. code-block:: python

    >>> with uproot.open("file.root:events") as tree:
    ...     do_something(tree)
    ...
    >>> with uproot.open("file.root")["events"] as tree:
    ...     do_something(tree)

Caches in Uproot 3 were strictly opt-in, but Uproot 4 provides a default: ``object_cache`` for extracted objects (histograms, TTrees) and ``array_cache`` for TTree data as arrays.

Removed features
""""""""""""""""

Uproot 4 does not have functions for specialized protocols like ``uproot3.http`` and ``uproot.xrootd``. Pass URLs with the appropriate scheme to the :func:`uproot.open <uproot.reading.open>` function.

Uproot 4 does not have specialized functions for reading data into Pandas DataFrames, like ``uproot3.pandas.iterate``. Use the normal :func:`uproot.iterate <uproot.behaviors.TBranch.iterate>` and :meth:`~uproot.behaviors.TBranch.TBranch.array`, :meth:`~uproot.behaviors.TBranch.HasBranches.arrays`, and :meth:`~uproot.behaviors.TBranch.HasBranches.iterate` functions with ``library="pd"`` to select Pandas as an output container.

Not yet implemented features
""""""""""""""""""""""""""""

Uproot 4 does not *yet* have an equivalent of ``uproot3.numentries`` (`#197 <https://github.com/scikit-hep/uproot4/issues/197>`__).

Uproot 4 cannot *yet* write data to files (`project 3 <https://github.com/scikit-hep/uproot4/projects/3>`__).

Internal differences
""""""""""""""""""""

* Remote sources are now read in exact byte ranges (Uproot 3 rounded to equal-sized chunks).

* All the byte ranges associated with a single call to :meth:`~uproot.behaviors.TBranch.HasBranches.arrays` are batched in a single request (HTTP multi-part GET or XRootD vector-read) to minimize the latency of requests.

Navigating a file
-----------------

Whereas Uproot 3 merged the functions of "file" and "directory," Uproot 4 has two distinct types: :class:`~uproot.reading.ReadOnlyFile` and :class:`~uproot.reading.ReadOnlyDirectory`. The :func:`uproot.open <uproot.reading.open>` function returns a :class:`~uproot.reading.ReadOnlyDirectory`, which is used to look up objects.

.. code-block:: python

    >>> directory = uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
    >>> directory
    <ReadOnlyDirectory '/' at 0x7c070dc03040>
    >>> directory["one/two/tree"]
    <TTree 'tree' (20 branches) at 0x78a2045fcf40>

The :class:`~uproot.reading.ReadOnlyFile`, on the other hand, is responsible for the physical :class:`~uproot.source.chunk.Source`, the ROOT file headers, streamers (``TStreamerInfo``), and therefore class definitions.

.. code-block:: python

    >>> directory.file
    <ReadOnlyFile 'https://scikit-hep.org/uproot3/examples/nesteddirs.root' at 0x7f6f27f85e80>
    >>> directory.file.uuid
    UUID('ac63575a-9ca4-11e7-9607-0100007fbeef')
    >>> directory.file.closed
    False
    >>> directory.file.show_streamers("TList")
    TString (v2)

    TObject (v1)
        fUniqueID: unsigned int (TStreamerBasicType)
        fBits: unsigned int (TStreamerBasicType)

    TCollection (v3): TObject (v1)
        fName: TString (TStreamerString)
        fSize: int (TStreamerBasicType)

    TSeqCollection (v0): TCollection (v3)

    TList (v5): TSeqCollection (v0)
    >>> directory.file.class_named("TTree")
    <class 'uproot.models.TTree.Model_TTree'>

Like Uproot 3, :class:`~uproot.reading.ReadOnlyDirectory` presents (and can accept) cycle numbers after a semicolon (``;``) and interprets slash (``/``) as a directory separator. Unlike Uproot 3, keys are presented as Python 3 ``str``, not ``bytes``, and the directory separator can extract a TTree's branches.

.. code-block:: python

    >>> directory = uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
    >>> directory
    <ReadOnlyDirectory '/' at 0x7c070dc03040>
    >>> directory.keys()
    ['one;1', 'one/two;1', 'one/two/tree;1', 'one/tree;1', 'three;1', 'three/tree;1']
    >>> directory["one"]
    <ReadOnlyDirectory '/one' at 0x78a2045f0fa0>
    >>> directory["one"]["two"]
    <ReadOnlyDirectory '/one/two' at 0x78a2045fcca0>
    >>> directory["one"]["two"]["tree"]
    <TTree 'tree' (20 branches) at 0x78a2045fcf40>
    >>> directory["one/two/tree"]
    <TTree 'tree' (20 branches) at 0x78a2045fcf40>
    >>> directory["three/tree/evt"]
    <TBranchElement 'evt' (39 subbranches) at 0x7f8cba86d880>
    >>> directory["three/tree/evt/I32"]
    <TBranchElement 'I32' at 0x7f8cba871f10>

In Uproot 3, directories could often be duck-typed as a mapping, but in Uproot 4, :class:`~uproot.reading.ReadOnlyDirectory` formally satisfies the ``Mapping`` protocol. As in Uproot 3, the :meth:`~uproot.reading.ReadOnlyDirectory.keys`, :meth:`~uproot.reading.ReadOnlyDirectory.values`, and :meth:`~uproot.reading.ReadOnlyDirectory.items` take options, but some defaults have changed:

* ``recursive=True`` is the new default (directories are recursively searched). There are no ``allkeys``, ``allvalues``, ``allitems`` methods for recursion.
* ``filter_name=None`` can be None, a string, a glob string, a regex in ``"/pattern/i"`` syntax, a function of str → bool, or an iterable of the above.
* ``filter_classname=None`` has the same options.

The ``filter_name`` and ``filter_classname`` mechanism is now uniform for :class:`~uproot.reading.ReadOnlyDirectory` and TTrees (:class:`~uproot.behaviors.TBranch.HasBranches`), though the latter is named ``filter_typename`` for TTrees.

Uproot 3's ``ROOTDirectory.classes`` and ``ROOTDirectory.allclasses``, which returned a list of 2-tuples of name, class object pairs, has become :meth:`~uproot.reading.ReadOnlyDirectory.classnames` in Uproot 4, which returns a dict mapping names to C++ class names.

.. code-block:: python

    >>> directory = uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
    >>> directory.classnames()
    {'one': 'TDirectory',
     'one/two': 'TDirectory',
     'one/two/tree': 'TTree',
     'one/tree': 'TTree',
     'three': 'TDirectory',
     'three/tree': 'TTree'}

This :meth:`~uproot.reading.ReadOnlyDirectory.classnames` method has the same options as :meth:`~uproot.reading.ReadOnlyDirectory.keys`, :meth:`~uproot.reading.ReadOnlyDirectory.values`, and :meth:`~uproot.reading.ReadOnlyDirectory.items`, but like :meth:`~uproot.reading.ReadOnlyDirectory.keys` (only), it doesn't initiate any data-reading.

To get the class object, use the :class:`~uproot.reading.ReadOnlyFile` or :meth:`~uproot.reading.ReadOnlyDirectory.class_of`.

.. code-block:: python

    >>> directory = uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
    >>> directory.file.class_named("TTree")
    <class 'uproot.models.TTree.Model_TTree'>
    >>> directory.class_of("one/two/tree")
    <class 'uproot.models.TTree.Model_TTree'>

In Uproot 4, requesting a class object *might* cause the file to read streamers (``TStreamerInfo``).

Examining TTrees
----------------

As in Uproot 3, TTrees have a :meth:`~uproot.behaviors.TBranch.HasBranches.show` method.

.. code-block:: python

    >>> tree = uproot.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root:three/tree")
    >>> tree.show()
    name                 | typename                 | interpretation
    ---------------------+--------------------------+-------------------------------
    evt                  | Event                    | AsGroup(<TBranchElement 'ev...
    evt/Beg              | TString                  | AsStrings()
    evt/I16              | int16_t                  | AsDtype('>i2')
    evt/I32              | int32_t                  | AsDtype('>i4')
    evt/I64              | int64_t                  | AsDtype('>i8')
    evt/U16              | uint16_t                 | AsDtype('>u2')
    evt/U32              | uint32_t                 | AsDtype('>u4')
    evt/U64              | uint64_t                 | AsDtype('>u8')
    evt/F32              | float                    | AsDtype('>f4')
    evt/F64              | double                   | AsDtype('>f8')
    evt/Str              | TString                  | AsStrings()
    evt/P3               | P3                       | AsGroup(<TBranchElement 'P3...
    evt/P3/P3.Px         | int32_t                  | AsDtype('>i4')
    evt/P3/P3.Py         | double                   | AsDtype('>f8')
    evt/P3/P3.Pz         | int32_t                  | AsDtype('>i4')
    evt/ArrayI16[10]     | int16_t[10]              | AsDtype("('>i2', (10,))")
    evt/ArrayI32[10]     | int32_t[10]              | AsDtype("('>i4', (10,))")
    evt/ArrayI64[10]     | int64_t[10]              | AsDtype("('>i8', (10,))")
    evt/ArrayU16[10]     | uint16_t[10]             | AsDtype("('>u2', (10,))")
    evt/ArrayU32[10]     | uint32_t[10]             | AsDtype("('>u4', (10,))")
    evt/ArrayU64[10]     | uint64_t[10]             | AsDtype("('>u8', (10,))")
    evt/ArrayF32[10]     | float[10]                | AsDtype("('>f4', (10,))")
    evt/ArrayF64[10]     | double[10]               | AsDtype("('>f8', (10,))")
    evt/N                | uint32_t                 | AsDtype('>u4')
    evt/SliceI16         | int16_t*                 | AsJagged(AsDtype('>i2'), he...
    evt/SliceI32         | int32_t*                 | AsJagged(AsDtype('>i4'), he...
    evt/SliceI64         | int64_t*                 | AsJagged(AsDtype('>i8'), he...
    evt/SliceU16         | uint16_t*                | AsJagged(AsDtype('>u2'), he...
    evt/SliceU32         | uint32_t*                | AsJagged(AsDtype('>u4'), he...
    evt/SliceU64         | uint64_t*                | AsJagged(AsDtype('>u8'), he...
    evt/SliceF32         | float*                   | AsJagged(AsDtype('>f4'), he...
    evt/SliceF64         | double*                  | AsJagged(AsDtype('>f8'), he...
    evt/StdStr           | std::string              | AsStrings(header_bytes=6)
    evt/StlVecI16        | std::vector<int16_t>     | AsJagged(AsDtype('>i2'), he...
    evt/StlVecI32        | std::vector<int32_t>     | AsJagged(AsDtype('>i4'), he...
    evt/StlVecI64        | std::vector<int64_t>     | AsJagged(AsDtype('>i8'), he...
    evt/StlVecU16        | std::vector<uint16_t>    | AsJagged(AsDtype('>u2'), he...
    evt/StlVecU32        | std::vector<uint32_t>    | AsJagged(AsDtype('>u4'), he...
    evt/StlVecU64        | std::vector<uint64_t>    | AsJagged(AsDtype('>u8'), he...
    evt/StlVecF32        | std::vector<float>       | AsJagged(AsDtype('>f4'), he...
    evt/StlVecF64        | std::vector<double>      | AsJagged(AsDtype('>f8'), he...
    evt/StlVecStr        | std::vector<std::string> | AsObjects(AsVector(True, As...
    evt/End              | TString                  | AsStrings()

However, this and other TTree-like behaviors are defined on a :class:`~uproot.behaviors.TBranch.HasBranches` class, which encompasses both :class:`~uproot.behaviors.TTree.TTree` and :class:`~uproot.behaviors.TBranch.TBranch`. This :meth:`~uproot.behaviors.TBranch.HasBranches` satisfies the ``Mapping`` protocol, and so do any nested branches:

.. code-block:: python

    >>> tree.keys()
    ['evt', 'evt/Beg', 'evt/I16', 'evt/I32', 'evt/I64', ..., 'evt/End']
    >>> tree["evt"].keys()
    ['Beg', 'I16', 'I32', 'I64', ..., 'End']

In addition to an :class:`~uproot.interpretation.Interpretation`, each :class:`~uproot.behaviors.TBranch.TBranch` also has a C++  :meth:`~uproot.behaviors.TBranch.TBranch.typename`, as shown above. Uproot 4 has a typename parser, and is able to interpret more types, including nested STL containers.

In addition to the standard ``Mapping`` methods, :meth:`~uproot.behaviors.TBranch.HasBranches.keys`, :meth:`~uproot.behaviors.TBranch.HasBranches.values`, and :meth:`~uproot.behaviors.TBranch.HasBranches.items`, :class:`~uproot.behaviors.TBranch.HasBranches` has a :meth:`~uproot.behaviors.TBranch.HasBranches.typenames` that returns str → str of branch names to their types. They have the same arguments:

* ``recursive=True`` is the new default (directories are recursively searched). There are no ``allkeys``, ``allvalues``, ``allitems`` methods for recursion.
* ``filter_name=None`` can be None, a string, a glob string, a regex in ``"/pattern/i"`` syntax, a function of str → bool, or an iterable of the above.
* ``filter_typename`` with the same options.
* ``filter_branch``, which can be None, :class:`~uproot.behaviors.TBranch.TBranch` → bool, :class:`~uproot.interpretation.Interpretation`, or None, to select by branch data.

Reading arrays
--------------

TTrees in Uproot 3 have ``array`` and ``arrays`` methods, which differ in how the resulting arrays are returned. In Uproot 4, :meth:`~uproot.behaviors.TBranch.TBranch.array` and :meth:`~uproot.behaviors.TBranch.HasBranches.arrays` have more differences:

* :meth:`~uproot.behaviors.TBranch.TBranch.array` is a :class:`~uproot.behaviors.TBranch.TBranch` method, but :meth:`~uproot.behaviors.TBranch.HasBranches.arrays` is a :class:`~uproot.behaviors.TBranch.HasBranches` method (which, admittedly, can overlap on a branch that has branches).
* :meth:`~uproot.behaviors.TBranch.HasBranches.arrays` can take a set of computable ``expressions`` and a ``cut``, but :meth:`~uproot.behaviors.TBranch.TBranch.array` never involves computation. The ``aliases`` and ``language`` arguments are also related to computation.
* Only :meth:`~uproot.behaviors.TBranch.TBranch.array` can override the default :class:`~uproot.interpretation.Interpretation` (it is the more low-level method).
* :meth:`~uproot.behaviors.TBranch.HasBranches.arrays` has the same ``filter_name``, ``filter_typename``, ``filter_branch`` as :meth:`~uproot.behaviors.TBranch.HasBranches.keys`. Since the ``expressions`` are now computable and glob wildcards (``*``) would be interpreted as multiplication, ``filter_name`` is the best way to select branches to read via a naming convention.

Some examples of simple reading and computing expressions:

.. code-block:: python

    >>> events = uproot.open("https://scikit-hep.org/uproot3/examples/Zmumu.root:events")

    >>> events["px1"].array()
    <Array [-41.2, 35.1, 35.1, ... 32.4, 32.5] type='2304 * float64'>

    >>> events.arrays(["px1", "py1", "pz1"])
    <Array [{px1: -41.2, ... pz1: -74.8}] type='2304 * {"px1": float64, "py1": float...'>

    >>> events.arrays("sqrt(px1**2 + py1**2)")
    <Array [{'sqrt(px1**2 + py1**2)': 44.7, ... ] type='2304 * {"sqrt(px1**2 + py1**...'>

    >>> events.arrays("pt1", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{pt1: 44.7}, ... {pt1: 32.4}] type='2304 * {"pt1": float64}'>

    >>> events.arrays(["M"], "pt1 > 50", aliases={"pt1": "sqrt(px1**2 + py1**2)"})
    <Array [{M: 91.8}, {M: 91.9, ... {M: 96.1}] type='290 * {"M": float64}'>

Some examples of filtering branches:

.. code-block:: python

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

In Uproot 3, you could specify whether the output is a dict of arrays, a tuple of arrays, or a Pandas DataFrame with the ``outputtype`` argument. In Uproot 4, these capabilities have been split into ``library`` and ``how``. The ``library`` determines which library will be used to represent the data that has been read. (You can also globally set ``uproot.default_library`` to avoid having to pass it to every ``arrays`` call.)

* ``library="np"`` to always return NumPy arrays (even ``dtype="O"`` if the type requires it);
* ``library="ak"`` (default) to always return Awkward Arrays;
* ``library="pd"`` to always return a Pandas Series or DataFrame.

(Uproot 3 chooses between NumPy and Awkward Array based on the type of the data. Since NumPy arrays and Awkward Arrays have different methods and properties, it's safer to write scripts with a deterministic output type.)

**Note:** Awkward Array is not one of Uproot 4's formal requirements. If you don't have ``awkward`` installed, :meth:`~uproot.behaviors.TBranch.TBranch.array` and :meth:`~uproot.behaviors.TBranch.HasBranches.arrays` will raise errors explaining how to install Awkward Array or switch to ``library="np"``. These errors might be hidden in automated testing, so be careful if you use that!

The ``how`` argument can be used to repackage arrays into dicts or tuples, and has special meanings for some libraries.

* For ``library="ak"``, passing ``how="zip"`` applies `ak.zip <https://awkward-array.readthedocs.io/en/latest/_auto/ak.zip.html>`__ to interleave data from compatible branches.
* For ``library="np"``, the ``how`` is passed to Pandas DataFrame merging.

Caching and parallel processing
-------------------------------

Uproot 3 and 4 both let you control caching by supplying any ``MutableMapping`` and parallel processing by supplying any Python 3 ``Executor``. What differs is the granularity of each.

Uproot 4 caching has less granularity. Other than objects,
