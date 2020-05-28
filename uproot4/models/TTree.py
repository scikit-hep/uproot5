# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import struct

# try:
#     from collections.abc import Mapping
# except ImportError:
#     from collections import Mapping

import uproot4.model
import uproot4.deserialization
import uproot4.models.TObject


_ttree_format1 = struct.Struct(">i")


# class Model_TTree(uproot4.model.Model, Mapping):
#     def read_members(self, chunk, cursor, context):
#         self._bases.append(
#             uproot4.models.TObject.Model_TObject.read(
#                 chunk, cursor, context, self._file, self._parent
#             )
#         )

#         self._members["fName"] = cursor.string(chunk)
#         self._members["fSize"] = cursor.field(chunk, _tlist_format1)


class Model_TTree(uproot4.model.DispatchByVersion):
    known_versions = {}


_tiofeatures_format1 = struct.Struct(">B")


class Model_ROOT_3a3a_TIOFeatures(uproot4.model.Model):
    def read_members(self, chunk, cursor, context):
        cursor.skip(4)
        self._members["fIOBits"] = cursor.field(chunk, _tiofeatures_format1)


uproot4.classes["TTree"] = Model_TTree
uproot4.classes["ROOT::TIOFeatures"] = Model_ROOT_3a3a_TIOFeatures
