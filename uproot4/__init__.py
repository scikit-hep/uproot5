# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

classes = {}
unknown_classes = {}

from uproot4.source.memmap import MemmapSource
from uproot4.source.file import FileSource
from uproot4.source.http import HTTPSource
from uproot4.source.http import MultithreadedHTTPSource
from uproot4.source.xrootd import XRootDSource
from uproot4.source.xrootd import MultithreadedXRootDSource
from uproot4.source.cursor import Cursor

from uproot4.reading import open
from uproot4.reading import no_filter
from uproot4.reading import ReadOnlyFile
from uproot4.reading import ReadOnlyDirectory

from uproot4.model import Model
from uproot4.model import classname_decode
from uproot4.model import classname_encode
from uproot4.model import has_class_named
from uproot4.model import class_named

import uproot4.models.TObject
import uproot4.models.TString
import uproot4.models.TNamed
import uproot4.models.TObjArray
import uproot4.models.TObjString
import uproot4.models.TList
import uproot4.models.THashList

# import uproot4.models.TRef
import uproot4.models.TArray

# import uproot4.models.ROOT_3a3a_TIOFeatures
import uproot4.models.RNTuple
