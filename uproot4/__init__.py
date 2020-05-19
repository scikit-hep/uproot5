# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

from uproot4.source.memmap import MemmapSource
from uproot4.source.file import FileSource
from uproot4.source.http import HTTPSource
from uproot4.source.http import MultithreadedHTTPSource
from uproot4.source.xrootd import XRootDSource
from uproot4.source.xrootd import MultithreadedXRootDSource

from uproot4.reading import open
from uproot4.reading import no_filter
from uproot4.reading import ReadOnlyFile
from uproot4.reading import ReadOnlyDirectory

streamers = {}
classes = {}
