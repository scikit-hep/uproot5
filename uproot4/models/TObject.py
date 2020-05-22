# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import uproot4.model
import uproot4.deserialization


class Class_TObject(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        uproot4.deserialization.skip_tobject(chunk, cursor, context)


uproot4.classes["TObject"] = Class_TObject
