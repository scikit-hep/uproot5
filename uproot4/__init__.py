# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

"""
Uproot: ROOT I/O in pure Python and NumPy.

Nearly all of the functions needed for general use are imported here, but the
documentation gives fully qualified names. For example, the most frequently
used function in Uproot is

.. code-block:: python

    uproot4.open("path/to/filename.root")

but we refer to it in the documentation as :doc:`uproot4.reading.open`.

Typical entry points for file-reading are

* :doc:`uproot4.reading.open`
* :doc:`uproot4.behaviors.TBranch.iterate`
* :doc:`uproot4.behaviors.TBranch.concatenate`
* :doc:`uproot4.behaviors.TBranch.lazy`

though they would usually be accessed as ``uproot4.iterate``,
``uproot4.concatenate``, and ``uproot4.lazy``.

The most useful classes are

* :doc:`uproot4.behaviors.TBranch.HasBranches` (``TTree`` or ``TBranch``)
* :doc:`uproot4.behaviors.TBranch.TBranch`
* :doc:`uproot4.behaviors.TH1`
* :doc:`uproot4.behaviors.TH2`
* :doc:`uproot4.behaviors.TProfile`

though they would usually be accessed through instances that have been read
from files.

The submodules of Uproot are:

* :doc:`uproot4.version`: for access to the version number.
* :doc:`uproot4.extras`: import functions for the libraries that Uproot can
  use, but does not require as dependencies. If a library can't be imported,
  these functions provide instructions for installing them.
* :doc:`uproot4.dynamic`: initially empty module, in which dynamically
  generated classes are defined.
* :doc:`uproot4.source`: the "physical layer," which reads bytes without
  interpreting them from various backends, like files, HTTP(S), and XRootD.
* :doc:`uproot4.compression`: functions for compressing and decompressing data.
* :doc:`uproot4.reading`: entry-point for reading files, as well as classes
  for the three basic types that can't be modeled: ``TFile``, ``TDirectory``,
  and ``TKey``.
* :doc:`uproot4.streamers`: models for ``TStreamerInfo`` and its elements
  to generate code for new models for classes in ROOT files.
* :doc:`uproot4.deserialization`: utility functions for deserialization,
  including the generation of new classes.
* :doc:`uproot4.model`: utilities for modeling C++ objects as Python objects.
* :doc:`uproot4.models`: predefined models for classes that are too basic
  to rely on ``TStreamerInfo`` or too common to justify reading it.
* :doc:`uproot4.behaviors`: methods and properties to mix into instantiated
  models, for a high-level user interface.
* :doc:`uproot4.interpretation`: prescriptions for converting ROOT types
  into Pythonic arrays.
* :doc:`uproot4.containers`: interpretations and models for standard
  containers, such as ``std::vector`` and arrays.
* :doc:`uproot4.compute`: computational backends for expressions in
  :doc:`uproot4.behavior.TBranch.HasBranches.arrays`.
* :doc:`uproot4.cache`: defines caches with least-recently used eviction
  policies.
* :doc:`uproot4.const`: integer constants used in ROOT serialization and
  deserialization.
* :doc:`uproot4._util`: non-public utilities used by any of the above.
"""

from __future__ import absolute_import

from uproot4.version import __version__

import uproot4.dynamic

classes = {}
unknown_classes = {}

from uproot4.cache import LRUCache
from uproot4.cache import LRUArrayCache

object_cache = LRUCache(100)
array_cache = LRUArrayCache("100 MB")

from uproot4.source.file import MemmapSource
from uproot4.source.file import MultithreadedFileSource
from uproot4.source.http import HTTPSource
from uproot4.source.http import MultithreadedHTTPSource
from uproot4.source.xrootd import XRootDSource
from uproot4.source.xrootd import MultithreadedXRootDSource
from uproot4.source.cursor import Cursor
from uproot4.source.futures import TrivialExecutor
from uproot4.source.futures import ThreadPoolExecutor

decompression_executor = ThreadPoolExecutor()
interpretation_executor = TrivialExecutor()

from uproot4.deserialization import DeserializationError

from uproot4.reading import open
from uproot4.reading import ReadOnlyFile
from uproot4.reading import ReadOnlyDirectory

from uproot4.model import Model
from uproot4.model import classname_decode
from uproot4.model import classname_encode
from uproot4.model import has_class_named
from uproot4.model import class_named
from uproot4.model import reset_classes

import uproot4.models.TObject
import uproot4.models.TString
import uproot4.models.TArray
import uproot4.models.TNamed
import uproot4.models.TList
import uproot4.models.THashList
import uproot4.models.TObjArray
import uproot4.models.TObjString
import uproot4.models.TAtt
import uproot4.models.TRef

import uproot4.models.TTree
import uproot4.models.TBranch
import uproot4.models.TLeaf
import uproot4.models.TBasket
import uproot4.models.RNTuple

from uproot4.containers import STLVector
from uproot4.containers import STLSet
from uproot4.containers import STLMap

import uproot4.interpretation
import uproot4.interpretation.library
from uproot4.interpretation.numerical import AsDtype
from uproot4.interpretation.numerical import AsDtypeInPlace
from uproot4.interpretation.numerical import AsDouble32
from uproot4.interpretation.numerical import AsFloat16
from uproot4.interpretation.numerical import AsSTLBits
from uproot4.interpretation.jagged import AsJagged
from uproot4.interpretation.strings import AsStrings
from uproot4.interpretation.objects import AsObjects
from uproot4.interpretation.objects import AsStridedObjects
from uproot4.interpretation.grouped import AsGrouped
from uproot4.containers import AsString
from uproot4.containers import AsPointer
from uproot4.containers import AsArray
from uproot4.containers import AsDynamic
from uproot4.containers import AsVector
from uproot4.containers import AsSet
from uproot4.containers import AsMap

default_library = "ak"

from uproot4.behaviors.TTree import TTree
from uproot4.behaviors.TBranch import TBranch
from uproot4.behaviors.TBranch import iterate
from uproot4.behaviors.TBranch import concatenate
from uproot4.behaviors.TBranch import lazy

import pkgutil
import uproot4.behaviors


def behavior_of(classname):
    """
    Finds and loads the behavior class for C++ (decoded) classname or returns
    None if there isn't one.

    Behaviors do not have a required base class, and they may be used with
    Awkward Array's ``ak.behavior``.

    The search strategy for finding behavior classes is:

    1. Translate the ROOT class name from C++ to Python with
       :doc:`uproot4.model.classname_encode`. For example,
       ``"ROOT::RThing"`` becomes ``"Model_ROOT_3a3a_RThing"``.
    2. Look for a submodule of ``uproot4.behaviors`` without
       the ``"Model_"`` prefix. For example, ``"ROOT_3a3a_RThing"``.
    3. Look for a class in that submodule with the fully encoded
       name. For example, ``"Model_ROOT_3a3a_RThing"``.

    See :doc:`uproot4.behaviors` for details.
    """
    name = classname_encode(classname)
    assert name.startswith("Model_")
    name = name[6:]

    if name not in globals():
        if name in behavior_of._module_names:
            exec(
                compile(
                    "import uproot4.behaviors.{0}".format(name), "<dynamic>", "exec"
                ),
                globals(),
            )
            module = eval("uproot4.behaviors.{0}".format(name))
            behavior_cls = getattr(module, name, None)
            if behavior_cls is not None:
                globals()[name] = behavior_cls

    return globals().get(name)


behavior_of._module_names = [
    module_name
    for loader, module_name, is_pkg in pkgutil.walk_packages(uproot4.behaviors.__path__)
]

del pkgutil


class KeyInFileError(KeyError):
    """
    Exception raised by attempts to find ROOT objects in ``TDirectories``
    or ``TBranches`` in :doc:`uproot4.behaviors.TBranch.HasBranches`, which
    both have a Python ``Mapping`` interface (square bracket syntax to extract
    items).

    This exception descends from Python's ``KeyError``, so it can be used in
    the normal way by interfaces that expect a missing item in a ``Mapping``
    to raise ``KeyError``, but it provides more information, depending on
    availability:

    * ``because``: an explanatory message
    * ``cycle``: the ROOT cycle number requested, if any
    * ``keys``: a list or partial list of keys that *are* in the object, in case
      of misspelling
    * ``file_path``: a path (or URL) to the file
    * ``object_path``: a path to the object within the ROOT file.
    """

    def __init__(
        self, key, because="", cycle=None, keys=None, file_path=None, object_path=None
    ):
        super(KeyInFileError, self).__init__(key)
        self.key = key
        self.because = because
        self.cycle = cycle
        self.keys = keys
        self.file_path = file_path
        self.object_path = object_path

    def __str__(self):
        if self.because == "":
            because = ""
        else:
            because = " because " + self.because

        with_keys = ""
        if self.keys is not None:
            to_show = None
            for key in self.keys:
                if to_show is None:
                    to_show = repr(key)
                else:
                    to_show += ", " + repr(key)
                if len(to_show) > 200:
                    to_show += "..."
                    break
            if to_show is None:
                to_show = "(none!)"
            with_keys = "\n\n    Known keys: {0}\n".format(to_show)

        in_file = ""
        if self.file_path is not None:
            in_file = "\nin file {0}".format(self.file_path)

        in_object = ""
        if self.object_path is not None:
            in_object = "\nin object {0}".format(self.object_path)

        if self.cycle == "any":
            return """not found: {0} (with any cycle number){1}{2}{3}{4}""".format(
                repr(self.key), because, with_keys, in_file, in_object
            )
        elif self.cycle is None:
            return """not found: {0}{1}{2}{3}{4}""".format(
                repr(self.key), because, with_keys, in_file, in_object
            )
        else:
            return """not found: {0} with cycle {1}{2}{3}{4}{5}""".format(
                repr(self.key), self.cycle, because, with_keys, in_file, in_object
            )


from uproot4._util import no_filter
