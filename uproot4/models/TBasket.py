# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

import uproot4.model
import uproot4.deserialization


class Model_TBasket(uproot4.model.Model):
    def read_numbytes_version(self, chunk, cursor, context):
        pass

    def read_members(self, chunk, cursor, context):
        cursor.debug(chunk, 80)
        raise Exception("STOP")


uproot4.classes["TBasket"] = Model_TBasket
