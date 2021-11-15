# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/main/LICENSE

"""
This module defines a versionless model for ``ROOT::Experimental::RNTuple``.
"""

from __future__ import absolute_import

import struct

try:
    import queue
except ImportError:
    import Queue as queue

import uproot

_rntuple_format1 = struct.Struct(">IIQIIQIIQ")

# https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#envelopes
_rntuple_frame_format = struct.Struct("<HHI")
_rntuple_feature_flag_format = struct.Struct("<Q")
_rntuple_num_bytes_fields = struct.Struct("<II")


class Model_ROOT_3a3a_Experimental_3a3a_RNTuple(uproot.model.Model):
    """
    A versionless :doc:`uproot.model.Model` for ``ROOT::Experimental::RNTuple``.
    """

    def read_members(self, chunk, cursor, context, file):
        if self.is_memberwise:
            raise NotImplementedError(
                """memberwise serialization of {0}
in file {1}""".format(
                    type(self).__name__, self.file.file_path
                )
            )

        cursor.skip(4)
        (
            self._members["fVersion"],
            self._members["fSize"],
            self._members["fSeekHeader"],
            self._members["fNBytesHeader"],
            self._members["fLenHeader"],
            self._members["fSeekFooter"],
            self._members["fNBytesFooter"],
            self._members["fLenFooter"],
            self._members["fReserved"],
        ) = cursor.fields(chunk, _rntuple_format1, context)

        seek, nbytes = self._members["fSeekHeader"], self._members["fNBytesHeader"]
        header_range = (seek, seek + nbytes)

        seek, nbytes = self._members["fSeekFooter"], self._members["fNBytesFooter"]
        footer_range = (seek, seek + nbytes)

        notifications = queue.Queue()
        compressed_header_chunk, compressed_footer_chunk = file.source.chunks(
            [header_range, footer_range], notifications=notifications
        )

        if self._members["fNBytesHeader"] == self._members["fLenHeader"]:
            self._header_chunk = compressed_header_chunk
            self._header_cursor = uproot.source.cursor.Cursor(
                self._members["fSeekHeader"]
            )
        else:
            self._header_chunk = uproot.compression.decompress(
                compressed_header_chunk,
                uproot.source.cursor.Cursor(self._members["fSeekHeader"]),
                context,
                self._members["fNBytesHeader"],
                self._members["fLenHeader"],
            )
            self._header_cursor = uproot.source.cursor.Cursor(0)

        if self._members["fNBytesFooter"] == self._members["fLenFooter"]:
            self._footer_chunk = compressed_footer_chunk
            self._footer_cursor = uproot.source.cursor.Cursor(
                self._members["fSeekFooter"]
            )
        else:
            self._footer_chunk = uproot.compression.decompress(
                compressed_footer_chunk,
                uproot.source.cursor.Cursor(self._members["fSeekFooter"]),
                context,
                self._members["fNBytesFooter"],
                self._members["fLenFooter"],
            )
            self._footer_cursor = uproot.source.cursor.Cursor(0)

        self._header, self._footer = None, None

    @property
    def header(self):
        if self._header is None:
            cursor = self._header_cursor.copy()
            context = {}

            self._header = {}
            self._header["frame"] = self._frame(self._header_chunk, cursor, context)

            # https://github.com/jblomer/root/blob/ntuple-binary-format-v1/tree/ntuple/v7/doc/specifications.md#header-envelope
            self._header["feature_flag"] = cursor.field(
                self._header_chunk, _rntuple_feature_flag_format, context
            )
            self._header["name"] = cursor.rntuple_string(self._header_chunk, context)
            self._header["description"] = cursor.rntuple_string(
                self._header_chunk, context
            )
            self._header["author"] = cursor.rntuple_string(self._header_chunk, context)

            cursor.skip(68)  # ???
            num_fields_plus_one = cursor.field(
                self._header_chunk, struct.Struct("<Q"), context
            )

            self._header["fields"] = [None] * (num_fields_plus_one - 1)
            while any(x is None for x in self._header["fields"]):
                field = {}

                pos = cursor.index
                field["num_bytes"] = cursor.field(
                    self._header_chunk, struct.Struct("<I"), context
                )
                field["id"] = cursor.field(
                    self._header_chunk, struct.Struct("<Q"), context
                )
                self._header["fields"][field["id"] - 1] = field

                cursor.skip(48)  # ???
                field["name"] = cursor.rntuple_string(self._header_chunk, context)
                field["description"] = cursor.rntuple_string(
                    self._header_chunk, context
                )
                field["type"] = cursor.rntuple_string(self._header_chunk, context)

                cursor.move_to(pos + field["num_bytes"])

        return self._header

    @property
    def footer(self):
        raise NotImplementedError

    def _frame(self, chunk, cursor, context):
        version, min_version, num_bytes = cursor.fields(
            chunk, _rntuple_frame_format, context
        )
        return {"version": version, "min_version": min_version, "num_bytes": num_bytes}


uproot.classes[
    "ROOT::Experimental::RNTuple"
] = Model_ROOT_3a3a_Experimental_3a3a_RNTuple
