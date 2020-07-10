# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

from uproot4.version import __version__

classes = {}
unknown_classes = {}

from uproot4.cache import LRUCache
from uproot4.cache import LRUArrayCache

from uproot4.source.memmap import MemmapSource
from uproot4.source.file import FileSource
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

default_library = "ak"

from uproot4.behaviors.TTree import TTree
from uproot4.behaviors.TBranch import TBranch
from uproot4.behaviors.TBranch import iterate
from uproot4.behaviors.TBranch import concatenate
from uproot4.behaviors.TBranch import lazy

import pkgutil
import uproot4.behaviors


def behavior_of(classname):
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
    __slots__ = ["key", "because", "cycle", "keys", "file_path", "object_path"]

    def __init__(
        self, key, because="", cycle=None, keys=None, file_path=None, object_path=None
    ):
        super(KeyInFileError, self).__init__(key)
        self.key = key
        self.because = because
        self.cycle = cycle
        self.file_path = file_path
        self.object_path = object_path
        self.keys = keys

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
