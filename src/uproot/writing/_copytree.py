# BSD 3-Clause License; see https://github.com/scikit-hep/uproot5/blob/main/LICENSE

"""
This is an internal module for writing TTrees in the "cascading" file writer. TTrees
are more like TDirectories than they are like histograms in that they can create
objects, TBaskets, which have to be allocated through the FreeSegments.

The implementation in this module does not use the TTree infrastructure in
:doc:`uproot.models.TTree`, :doc:`uproot.models.TBranch`, and :doc:`uproot.models.TBasket`,
since the models intended for reading have to adapt to different class versions, but
a writer can always write the same class version, and because writing involves allocating
and sometimes freeing data.

See :doc:`uproot.writing._cascade` for a general overview of the cascading writer concept.
"""
from __future__ import annotations

import datetime
import math
import struct
import warnings
from collections.abc import Mapping

import numpy

import uproot.compression
import uproot.const
import uproot.reading
import uproot.serialization
# from uproot.writing.writable import __setitem__

_dtype_to_char = {
    numpy.dtype("bool"): "O",
    numpy.dtype(">i1"): "B",
    numpy.dtype(">u1"): "b",
    numpy.dtype(">i2"): "S",
    numpy.dtype(">u2"): "s",
    numpy.dtype(">i4"): "I",
    numpy.dtype(">u4"): "i",
    numpy.dtype(">i8"): "L",
    numpy.dtype(">u8"): "l",
    numpy.dtype(">f4"): "F",
    numpy.dtype(">f8"): "D",
    numpy.dtype(">U"): "C",
}


class Tree:
    """
    Writes a TTree, including all TBranches, TLeaves, and (upon ``extend``) TBaskets.

    Rather than treating TBranches as a separate object, this *writable* TTree writes
    the whole metadata block in one function, so that interrelationships are easier
    to preserve.

    Writes the following class instance versions:

    - TTree: version 20
    - TBranch: version 13
    - TLeaf: version 2
    - TLeaf*: version 1
    - TBasket: version 3

    The ``write_anew`` method writes the whole tree, possibly for the first time, possibly
    because it has been moved (exceeded its initial allocation of TBasket pointers).

    The ``write_updates`` method rewrites the parts that change when new TBaskets are
    added.

    The ``extend`` method adds a TBasket to every TBranch.

    The ``write_np_basket`` and ``write_jagged_basket`` methods write one TBasket in one
    TBranch, either a rectilinear one from NumPy or a simple jagged array from Awkward Array.

    See `ROOT TTree specification <https://github.com/root-project/root/blob/master/io/doc/TFile/ttree.md>`__.
    """

    def __init__(
        self,
        directory,
        freesegments,
        source,
        name, # new name
        file,
        # new_branch,
    ):
        # 1: are any of these actually attributes of whatever type source will end up being
        # 2: so would source already be decompressed? Does any of this work? readonlyttree?
        # Use "readonlykey" to get
        self.source = source
        self._directory = directory # good
        self._name = name
        self._title = source.title
        self._freesegments = freesegments
        self._file = file
        # self._counter_name
        # self._field_name = source.field_name
        # self._basket_capacity = source.initial_basket_capacity
        # self._resize_factor = source.resize_factor

        self._num_entries = source.num_entries
        self._num_baskets = 0

        self._metadata_start = None
        self._metadata = {
            "fTotBytes": source.members["fTotBytes"],
            "fZipBytes": source.members["fZipBytes"],
            "fSavedBytes": source.members["fSavedBytes"],
            "fFlushedBytes": source.members["fFlushedBytes"],
            "fWeight": source.members["fWeight"],
            "fTimerInterval": source.members["fTimerInterval"],
            "fScanField": source.members["fScanField"],
            "fUpdate": source.members["fUpdate"],
            "fDefaultEntryOffsetLen": source.members["fDefaultEntryOffsetLen"],
            "fNClusterRange": source.members["fNClusterRange"],
            "fMaxEntries": source.members["fMaxEntries"],
            "fMaxEntryLoop": source.members["fMaxEntryLoop"],
            "fMaxVirtualSize": source.members["fMaxVirtualSize"],
            "fAutoSave": source.members["fAutoSave"],
            "fAutoFlush": source.members["fAutoFlush"],
            "fEstimate": source.members["fEstimate"],

        }

        self._key = None


        # Add new branch (eventually!)
        # self._branch_data.append({"kind":"record, "})

        # for branch_name, branch_type in source.branches:
        #     branch_dict = None
        #     branch_dtype = None
        #     branch_datashape = None

        # self.__setitem__() # where?
        self._branch_data = []
        for branch in source.branches:
            self._branch_data.append({key: data for key, data in branch.all_members.items()})
            self._branch_data[-1].update(
                {
                    "arrays_write_start": 0,
                    "arrays_write_stop": 0,
                    "metadata_start": None,
                    "basket_metadata_start": None,
                    "tleaf_reference_number": None,
                    "tleaf_maximum_value": 0,
                    "tleaf_special_struct": None,
                }
            )
            self._num_baskets += branch.num_baskets
    # def __repr__(self):
    #     return "{}({}, {}, {}, {}, {}, {}, {})".format(
    #         type(self).__name__,
    #         self._directory,
    #         self._name,
    #         self._title,
    #         [(datum["fName"], datum["branch_type"]) for datum in self._branch_data],
    #         self._freesegments,
    #         self._basket_capacity,
    #         self._resize_factor,
    #     )

    def write_copy(self, sink):
        key_num_bytes = uproot.reading._key_format_big.size + 6
        name_asbytes = self._name.encode(errors="surrogateescape")
        title_asbytes = self._title.encode(errors="surrogateescape")
        key_num_bytes += (1 if len(name_asbytes) < 255 else 5) + len(name_asbytes)
        key_num_bytes += (1 if len(title_asbytes) < 255 else 5) + len(title_asbytes)

        out = [None]
        ttree_header_index = 0

        tobject = uproot.models.TObject.Model_TObject.empty()
        tnamed = uproot.models.TNamed.Model_TNamed.empty()
        tnamed._bases.append(tobject)
        tnamed._members["fTitle"] = self._title
        tnamed._serialize(out, True, self._name, uproot.const.kMustCleanup)

        # TAttLine v2, fLineColor: 602 fLineStyle: 1 fLineWidth: 1
        # TAttFill v2, fFillColor: 0, fFillStyle: 1001
        # TAttMarker v2, fMarkerColor: 1, fMarkerStyle: 1, fMarkerSize: 1.0
        out.append(
            b"@\x00\x00\x08\x00\x02\x02Z\x00\x01\x00\x01"
            b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9"
            b"@\x00\x00\n\x00\x02\x00\x01\x00\x01?\x80\x00\x00"
        )
        metadata_out_index = len(out)
        out.append(
            uproot.models.TTree._ttree20_format1.pack(
                self.source.members["fEntries"],
                self._metadata["fTotBytes"],
                self._metadata["fZipBytes"],
                self._metadata["fSavedBytes"],
                self._metadata["fFlushedBytes"],
                self._metadata["fWeight"],
                self._metadata["fTimerInterval"],
                self._metadata["fScanField"],
                self._metadata["fUpdate"],
                self._metadata["fDefaultEntryOffsetLen"],
                self._metadata["fNClusterRange"],
                self._metadata["fMaxEntries"],
                self._metadata["fMaxEntryLoop"],
                self._metadata["fMaxVirtualSize"],
                self._metadata["fAutoSave"],
                self._metadata["fAutoFlush"],
                self._metadata["fEstimate"],
            )
        )

        # speedbump (0), fClusterRangeEnd (empty array),
        # speedbump (0), fClusterSize (empty array)
        # fIOFeatures (TIOFeatures)
        out.append(b"\x00\x00@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

        tleaf_reference_numbers = []

        tobjarray_of_branches_index = len(out)
        out.append(None)

        # TObjArray header with fName: ""
        out.append(b"\x00\x01\x00\x00\x00\x00\x03\x00@\x00\x00")
        out.append(
            uproot.models.TObjArray._tobjarray_format1.pack(
                len(self.source.members['fBranches']),  # TObjArray fSize
                0,  # TObjArray fLowerBound
            )
        )
        for branch_indx, _value in enumerate(self._branch_data):
            datum = self._branch_data[branch_indx]
            # if datum["kind"] == "record":
            #     continue

            any_tbranch_index = len(out)
            out.append(None)
            out.append(b"TBranch\x00")

            tbranch_index = len(out)
            out.append(None)

            tbranch_tobject = uproot.models.TObject.Model_TObject.empty()
            tbranch_tnamed = uproot.models.TNamed.Model_TNamed.empty()
            tbranch_tnamed._bases.append(tbranch_tobject)
            tbranch_tnamed._members["fTitle"] = datum["fTitle"]
            tbranch_tnamed._serialize(
                out, True, datum["fName"], numpy.uint32(0x00400000)
            )
            # TAttFill v2, fFillColor: 0, fFillStyle: 1001
            out.append(b"@\x00\x00\x06\x00\x02\x00\x00\x03\xe9")

            assert sum(1 if x is None else 0 for x in out) == 4
            datum["metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # Lie about the compression level so that ROOT checks and does the right thing.
            # https://github.com/root-project/root/blob/87a998d48803bc207288d90038e60ff148827664/tree/tree/src/TBasket.cxx#L560-L578
            # Without this, when small buffers are left uncompressed, ROOT complains about them not being compressed.
            # (I don't know where the "no, really, this is uncompressed" bit is.)

            # print(self.source.branches[branch_indx].member("fCompress"))
            out.append(
                uproot.models.TBranch._tbranch13_format1.pack(
                    self.source.branches[branch_indx].member("fCompress"),
                    datum["fBasketSize"],
                    datum["fEntryOffsetLen"],
                    self.source.branches[branch_indx].member("fWriteBasket"),  # fWriteBasket
                    self._num_entries,  # fEntryNumber
                )
            )

            # fIOFeatures (TIOFeatures)
            out.append(b"@\x00\x00\x07\x00\x00\x1a\xa1/\x10\x00")

            out.append(
                uproot.models.TBranch._tbranch13_format2.pack(
                    datum["fOffset"],
                    datum["fMaxBaskets"],  # fMaxBaskets
                    datum["fSplitLevel"],
                    datum["fEntries"],  # fEntries
                    datum["fFirstEntry"],
                    datum["fTotBytes"],
                    datum["fZipBytes"],
                )
            )

            # empty TObjArray of TBranches
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            subtobjarray_of_leaves_index = len(out)
            out.append(None)

            # TObjArray header with fName: "", fSize: 1, fLowerBound: 0
            out.append(
                b"\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00"
            )

            absolute_location = key_num_bytes + sum(
                len(x) for x in out if x is not None
            )
            absolute_location += 8 + 6 * (sum(1 if x is None else 0 for x in out) - 1)
            datum["tleaf_reference_number"] = absolute_location + 2
            tleaf_reference_numbers.append(datum["tleaf_reference_number"])

            subany_tleaf_index = len(out)
            out.append(None)

            # Leaf stuff is missing, def part of the problem?
            encoded_classname = self.source.branches[branch_indx].member("fLeaves")[0].encoded_classname
            if encoded_classname == "Model_TLeafB_v1":
                letter_upper = "B"
                special_struct = uproot.models.TLeaf._tleafb1_format1
            elif encoded_classname == "Model_TLeafS_v1":
                letter_upper = "S"
                special_struct = uproot.models.TLeaf._tleafs1_format1
            elif encoded_classname == "Model_TLeafI_v1":
                letter_upper = "I"
                special_struct = uproot.models.TLeaf._tleafi1_format1
            elif encoded_classname == "Model_TLeafL_v1":
                letter_upper = "L"
                special_struct = uproot.models.TLeaf._tleafl1_format0
            elif encoded_classname == "Model_TLeafF_v1":
                letter_upper = "F"
                special_struct = uproot.models.TLeaf._tleaff1_format1
            elif encoded_classname == "Model_TLeafD_v1":
                letter_upper = "D"
                special_struct = uproot.models.TLeaf._tleafd1_format1
            elif encoded_classname == "Model_TLeafC_v1":
                letter_upper = "C"
                special_struct = uproot.models.TLeaf._tleafc1_format1
            else:
                letter_upper = "O"
                special_struct = uproot.models.TLeaf._tleafO1_format1

            out.append(("TLeaf" + letter_upper).encode() + b"\x00")

            # if self._branch_data[branch_indx]["shape"] == ():
            dims = ""
            # else:
            #     dims = "".join("[" + str(x) + "]" for x in self._branch_data[branch_indx]["shape"])
            if self.source.branches[branch_indx].count_branch is not None:
                dims = "[" + self.source.branches[branch_indx].count_branch.name + "]" + dims

            # single TLeaf
            leaf_name = datum["fName"].encode(errors="surrogateescape")
            leaf_title = (datum["fName"]).encode(errors="surrogateescape")
            leaf_name_length = (1 if len(leaf_name) < 255 else 5) + len(leaf_name)
            leaf_title_length = (1 if len(leaf_title) < 255 else 5) + len(leaf_title)

            leaf_header = numpy.array(
                [
                    64,
                    0,
                    0,
                    76,
                    0,
                    1,
                    64,
                    0,
                    0,
                    54,
                    0,
                    2,
                    64,
                    0,
                    0,
                    30,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    3,
                    0,
                    0,
                    0,
                ],
                numpy.uint8,
            )
            tmp = leaf_header[0:4].view(">u4")
            tmp[:] = (
                numpy.uint32(
                    42 + leaf_name_length + leaf_title_length + special_struct.size
                )
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[6:10].view(">u4")
            tmp[:] = (
                numpy.uint32(36 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )
            tmp = leaf_header[12:16].view(">u4")
            tmp[:] = (
                numpy.uint32(12 + leaf_name_length + leaf_title_length)
                | uproot.const.kByteCountMask
            )

            out.append(uproot._util.tobytes(leaf_header))
            if len(leaf_name) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_name), len(leaf_name), leaf_name)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_name), 255, len(leaf_name), leaf_name
                    )
                )
            if len(leaf_title) < 255:
                out.append(
                    struct.pack(">B%ds" % len(leaf_title), len(leaf_title), leaf_title)
                )
            else:
                out.append(
                    struct.pack(
                        ">BI%ds" % len(leaf_title), 255, len(leaf_title), leaf_title
                    )
                )

            # fLen = 1
            # for item in datum["shape"]:
            #     fLen *= item

            # generic TLeaf members
            out.append(
                uproot.models.TLeaf._tleaf2_format0.pack(
                    self.source.branches[branch_indx].member("fLeaves")[0].member("fLen"),
                    self.source.branches[branch_indx].member("fLeaves")[0].member("fLenType"),
                    self.source.branches[branch_indx].member("fLeaves")[0].member("fOffset"),  # fOffset
                    self.source.branches[branch_indx].member("fLeaves")[0].member("fIsRange"),  # fIsRange
                    self.source.branches[branch_indx].member("fLeaves")[0].member("fIsUnsigned"),
                )
            )
            if self.source.branches[branch_indx].member("fLeaves")[0].member("fLeafCount") is None:
                # null fLeafCount
                out.append(b"\x00\x00\x00\x00")
            else:
                # reference to fLeafCount
                print("fingers crossed", self.source.branches[branch_indx].member("fLeaves")[0].member("fLeafCount").all_members)
                out.append(
                    uproot.deserialization._read_object_any_format1.pack(
                        datum["counter"]["tleaf_reference_number"]
                    )
                )

            # # specialized TLeaf* members (fMinimum, fMaximum)
            out.append(special_struct.pack(0, 0))
            datum["tleaf_special_struct"] = special_struct

            out[subany_tleaf_index] = (
                uproot.serialization._serialize_object_any_format1.pack(
                    numpy.uint32(sum(len(x) for x in out[subany_tleaf_index + 1 :]) + 4)
                    | uproot.const.kByteCountMask,
                    uproot.const.kNewClassTag,
                )
            )

            out[subtobjarray_of_leaves_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[subtobjarray_of_leaves_index + 1 :]),
                3,  # TObjArray
            )

            # empty TObjArray of fBaskets (embedded)
            out.append(
                b"@\x00\x00\x15\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
            )

            assert sum(1 if x is None else 0 for x in out) == 4
            self._branch_data[branch_indx]["basket_metadata_start"] = (6 + 6 + 8 + 6) + sum(
                len(x) for x in out if x is not None
            )

            # speedbump and fBasketBytes
            out.append(b"\x01")
            out.append(uproot._util.tobytes(self._branch_data[branch_indx]["fBasketBytes"]))
            # speedbump and fBasketEntry
            out.append(b"\x01")
            out.append(uproot._util.tobytes(self._branch_data[branch_indx]["fBasketEntry"]))

            # speedbump and fBasketSeek
            out.append(b"\x01")
            out.append(uproot._util.tobytes(self._branch_data[branch_indx]["fBasketSeek"]))

            # empty fFileName
            out.append(b"\x00")
            out[tbranch_index] = uproot.serialization.numbytes_version(
                sum(len(x) for x in out[tbranch_index + 1 :]), 13  # TBranch
            )

            out[any_tbranch_index] = (
                uproot.serialization._serialize_object_any_format1.pack(
                    numpy.uint32(sum(len(x) for x in out[any_tbranch_index + 1 :]) + 4)
                    | uproot.const.kByteCountMask,
                    uproot.const.kNewClassTag,
                )
            )
        out[tobjarray_of_branches_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[tobjarray_of_branches_index + 1 :]), 3  # TObjArray
        )

        # TObjArray of TLeaf references
        tleaf_reference_bytes = uproot._util.tobytes(
            numpy.array(tleaf_reference_numbers, ">u4")
        )
        out.append(
            struct.pack(
                ">I13sI4s",
                (21 + len(tleaf_reference_bytes)) | uproot.const.kByteCountMask,
                b"\x00\x03\x00\x01\x00\x00\x00\x00\x03\x00\x00\x00\x00",
                len(tleaf_reference_numbers),
                b"\x00\x00\x00\x00",
            )
        )

        out.append(tleaf_reference_bytes)

        # null fAliases (b"\x00\x00\x00\x00")
        # empty fIndexValues array (4-byte length is zero)
        # empty fIndex array (4-byte length is zero)
        # null fTreeIndex (b"\x00\x00\x00\x00")
        # null fFriends (b"\x00\x00\x00\x00")
        # null fUserInfo (b"\x00\x00\x00\x00")
        # null fBranchRef (b"\x00\x00\x00\x00")
        out.append(b"\x00" * 28)

        out[ttree_header_index] = uproot.serialization.numbytes_version(
            sum(len(x) for x in out[ttree_header_index + 1 :]), 20  # TTre[e
        )
        self._metadata_start = sum(len(x) for x in out[:metadata_out_index])

        raw_data = b"".join(out)
        self._key = self._directory.add_object(
            sink,
            "TTree",
            self._name,
            self._title,
            raw_data,
            len(raw_data),
            replaces=self._key,
            big=True,
        )

        # actually write baskets!!??
        # if self._num_baskets >= self._basket_capacity - 1:
        #     self._basket_capacity = max(
        #         self._basket_capacity + 1,
        #         int(math.ceil(self._basket_capacity * self._resize_factor)),
        #     )

        for datum in self._branch_data:
            # fBasketBytes = datum["fBasketBytes"]
            # fBasketEntry = datum["fBasketEntry"]
            # fBasketSeek = datum["fBasketSeek"]
            # datum["fBasketBytes"][: len(fBasketBytes)] = fBasketBytes
            # datum["fBasketEntry"][: len(fBasketEntry)] = fBasketEntry
            # datum["fBasketSeek"][: len(fBasketSeek)] = fBasketSeek
            # datum["fBasketEntry"][len(fBasketEntry)] = self._num_entries

        oldloc = start = self._key.location
        stop = start + self._key.num_bytes + self._key.compressed_bytes
        self.write_anew(sink)

        newloc = self._key.seek_location
        self.file._move_tree(oldloc, newloc)

        self._freesegments.release(start, stop)
        sink.set_file_length(self._freesegments.fileheader.end)
        sink.flush()



def get_counter_branches(self):
    """
    Gets counter branches to remove them in merge etc.
    """
    import numpy as np
    count_branches = []
    for branch in self.source.keys():  # noqa: SIM118
        if self.source[branch].member("fLeaves")[0].member("fLeafCount") is None:
            continue
        count_branches.append(self.source[branch].count_branch.name)
    return np.unique(count_branches, axis=0)
