# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4._util
import uproot4.source.cursor
import uproot4.source.chunk
import uproot4.source.file
import uproot4.source.memmap
import uproot4.source.http
import uproot4.source.xrootd


class ReadOnlyFile(object):
    defaults = {
        "file_handler": uproot4.source.memmap.MemmapSource,
        "xrootd_handler": uproot4.source.xrootd.XRootDSource,
        "http_handler": uproot4.source.http.HTTPSource,
        "timeout": 30,
        "max_num_elements": None,
        "num_workers": 10,
        "num_fallback_workers": 10,
    }

    def __init__(self, file_path, **options):
        all_options = dict(self.defaults)
        all_options.update(options)

        Source = uproot4._util.path_to_source_class(file_path, options)
        self._source = Source(file_path, **options)

    @property
    def source(self):
        return self._source
