# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import re

__version__ = "0.0.14"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re
