Uproot 3 → 4 cheat-sheet
========================

The Uproot 3 → 4 transition was primarily motivated by Awkward Array 0 → 1. The interface of Awkward Array significantly changed and Awkward Arrays are output by Uproot functions, so this difference would be visible to you as a user of Uproot.

Thus, it was also a good time to introduce interface changes to Uproot itself—such as presenting C++ strings as Python 3 ``str``, rather than uninterpreted ``bytes``.

Fundamental changes were also required to streamline remote reading (HTTP and XRootD), so Uproot 4 was distributed as a separate project in parallel with Uproot 3 (like Awkward 1 and 0). For the latter half of 2020, adventurous users and downstream developers could install `uproot4 <https://pypi.org/project/uproot4/>`__ as a separate project.

.. image:: https://raw.githubusercontent.com/scikit-hep/uproot4/jpivarski/write-cheat-sheet/docs-img/diagrams/uproot-awkward-timeline.png
  :width: 100%

On December 1, 2020, however, Awkward 0 and Uproot 3 were deprecated, moved to PyPI packages `awkward0 <https://pypi.org/project/awkward0/>`__ and `uproot3 <https://pypi.org/project/uproot3/>`__, while Awkward 1 and Uproot 4 became unqualified as `awkward <https://pypi.org/project/awkward/>`__ and `uproot <https://pypi.org/project/uproot/>`__.

This document is to help users of Uproot 3 get started on Uproot 4.

Opening a file
--------------

The "open" function is still named :func:`uproot4.open <uproot4.reading.open>`, and it still recognizes local files, HTTP, and XRootD by the URL prefix (or lack thereof).

.. code-block:: python

    >>> import uproot4
    >>> local_file = uproot4.open("local/file.root")
    >>> http_file = uproot4.open("https://server.net/file.root")
    >>> xrootd_file = uproot4.open("root://server.net/file.root")

But whereas Uproot 3 took a handler (local/HTTP/XRootD) and options as a class instance, handlers are specified in Uproot 4 by passing a class and options are free-standing arguments.

.. code-block:: python

    >>> file = uproot4.open("file.root",
    ...                     file_handler=uproot4.MultithreadedFileSource,
    ...                     num_workers=10)
    >>> file = uproot4.open("https://server.net/file.root",
    ...                     http_handler=uproot4.MultithreadedHTTPSource,
    ...                     timeout=3.0)
    >>> file = uproot4.open("root://server.net/file.root",
    ...                     xrootd_handler=uproot4.MultithreadedXRootDSource,
    ...                     num_workers=5)

As in Uproot 3, there is a function for iterating over many files, :func:`uproot4.iterate <uproot4.behaviors.TBranch.iterate>`, and for lazily opening files, :func:`uproot4.lazy <uproot4.behaviors.TBranch.lazy>` (replacing ``uproot3.lazyarray`` and ``uproot3.lazyarrays``). Uproot 4 has an additional function for reading many files into one array (not lazily): :func:`uproot4.concatenate <uproot4.behaviors.TBranch.concatenate>`.

Array-reading differences are covered in `Reading arrays <#reading-arrays>`__, below. File-opening differences are illustrated well enough with :func:`uproot4.open <uproot4.reading.open>`.

New features
""""""""""""

Files can now truly be closed (long story), so the ``with`` syntax is recommended for scripts that open a lot of files.

.. code-block:: python

    >>> with uproot4.open("file.root") as file:
    ...     do_something(file)

Python file objects can be passed to :func:`uproot4.open <uproot4.reading.open>` in place of a filename string.

.. code-block:: python

    >>> import tarfile
    >>> with tarfile.open("file.tar.gz", "r") as tar:
    ...     with uproot4.open(tar) as file:
    ...         do_something(file)

There's a filename syntax for opening a file and pulling one object out of it. This is primarily for convenience but was strongly requested (`#79 <https://github.com/scikit-hep/uproot4/issues/79>`__).

.. code-block:: python

    >>> histogram = uproot4.open("file.root:path/to/histogram")

So what if the filename has a colon (``:``) in it? (Note: URLs are properly handled.) You have two options: (1) ``pathlib.Path`` objects are never parsed for the colon separator and (2) you can also express the separation with a dict.

.. code-block:: python

    >>> histogram = uproot4.open(pathlib.Path("real:colon.root"))["histogram"]
    >>> histogram = uproot4.open({"real:colon.root": "histogram"})

Error messages about missing files will remind you of the options.

If you want to use this with a context manager (``with`` statement), closing the extracted object closes the file it came from.

.. code-block:: python

    >>> with uproot4.open("file.root:events") as tree:
    ...     do_something(tree)
    ... 
    >>> with uproot4.open("file.root")["events"] as tree:
    ...     do_something(tree)

Caches in Uproot 3 were strictly opt-in, but Uproot 4 provides a default: ``object_cache`` for extracted objects (histograms, TTrees) and ``array_cache`` for TTree data as arrays.

Removed features
""""""""""""""""

Uproot 4 does not have functions for specialized protocols like ``uproot3.http`` and ``uproot4.xrootd``. Pass URLs with the appropriate scheme to the :func:`uproot4.open <uproot4.reading.open>` function.

Uproot 4 does not have specialized functions for reading data into Pandas DataFrames, like ``uproot3.pandas.iterate``. Use the normal :func:`uproot4.iterate <uproot4.behaviors.TBranch.iterate>` and :meth:`~uproot4.behaviors.TBranch.TBranch.array`, :meth:`~uproot4.behaviors.TBranch.HasBranches.arrays`, and :meth:`~uproot4.behaviors.TBranch.HasBranches.iterate` functions with ``library="pd"`` to select Pandas as an output container.

Not yet implemented features
""""""""""""""""""""""""""""

Uproot 4 does not *yet* have an equivalent of ``uproot3.numentries`` (`#197 <https://github.com/scikit-hep/uproot4/issues/197>`__).

Uproot 4 cannot *yet* write data to files (`project 3 <https://github.com/scikit-hep/uproot4/projects/3>`__).

Internal differences
""""""""""""""""""""

* Remote sources are now read in exact byte ranges (Uproot 3 rounded to equal-sized chunks).

* All the byte ranges associated with a single call to :meth:`~uproot4.behaviors.TBranch.HasBranches.arrays` are batched in a single request (HTTP multi-part GET or XRootD vector-read) to minimize the latency of requests.

Navigating a file
-----------------

Whereas Uproot 3 merged the functions of "file" and "directory," Uproot 4 has two distinct types: :class:`~uproot4.reading.ReadOnlyFile` and :class:`~uproot4.reading.ReadOnlyDirectory`. The :func:`uproot4.open <uproot4.reading.open>` function returns a :class:`~uproot4.reading.ReadOnlyDirectory`, which is used to look up objects.

.. code-block:: python

    >>> directory = uproot4.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
    >>> directory
    <ReadOnlyDirectory '/' at 0x7c070dc03040>
    >>> directory["one/two/tree"]
    <TTree 'tree' (20 branches) at 0x78a2045fcf40>

The :class:`~uproot4.reading.ReadOnlyFile`, on the other hand, is responsible for the physical :class:`~uproot4.source.chunk.Source`, the ROOT file headers, streamers (``TStreamerInfo``), and therefore class definitions.

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
    <class 'uproot4.models.TTree.Model_TTree'>

Like Uproot 3, :class:`~uproot4.reading.ReadOnlyDirectory` presents (and can accept) cycle numbers after a semicolon (``;``) and interprets slash (``/``) as a directory separator. Unlike Uproot 3, keys are presented as Python 3 ``str``, not ``bytes``, and the directory separator can extract a TTree's branches.

.. code-block:: python

    >>> directory = uproot4.open("https://scikit-hep.org/uproot3/examples/nesteddirs.root")
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
    >>> directory["one/two/tree/ArrayInt64"]
    <TBranch 'ArrayInt64' at 0x7f6f27d178e0>

In Uproot 3, directories could often be duck-typed as a mutable mapping, but in Uproot 4, :class:`~uproot4.reading.ReadOnlyDirectory` formally satisfies the ``MutableMapping`` protocol. As in Uproot 3, the :meth:`~uproot4.reading.ReadOnlyDirectory.keys`, :meth:`~uproot4.reading.ReadOnlyDirectory.values`, and :meth:`~uproot4.reading.ReadOnlyDirectory.items` take options, but some defaults have changed:

* ``recursive=True`` is the new default (directories are recursively searched). There are no ``allkeys``, ``allvalues``, ``allitems`` methods.
* ``filter_name=None`` can be None, a string, a glob string, a regex in ``"/pattern/i"`` syntax, a function of str → bool, or an iterable of the above.
* ``filter_classname=None`` has the same options.

The ``filter_name`` and ``filter_classname`` mechanism is now uniform for :class:`~uproot4.reading.ReadOnlyDirectory` and TTrees (:class:`~uproot4.behaviors.TBranch.HasBranches`), though the latter is named ``filter_typename`` for TTrees.

Uproot 3's 



Reading arrays
--------------
